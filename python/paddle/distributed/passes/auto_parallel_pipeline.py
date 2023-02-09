# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import exception
import os

from paddle.fluid import core
from .pass_base import PassBase, register_pass
from paddle.fluid.framework import Program, Parameter
from paddle.distributed.fleet.fleet_executor_utils import TaskNode
from paddle.distributed.fleet.meta_optimizers.common import OpRole

from paddle.distributed.auto_parallel.utils import (
    is_forward_op,
    is_backward_op,
    is_optimize_op,
    is_lr_sched_op,
)


__not_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]


@register_pass("auto_parallel_pipeline")
class PipelinePass(PassBase):
    def __init__(self):
        super(PipelinePass, self).__init__()
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self._dist_context = self.get_attr("dist_context")
        self._acc_steps = self.get_attr("accumulate_steps")
        self._mode = self.get_attr("schedule_mode")
        self._gen_bsz = self.get_attr("generation_batch_size")
        self._program = main_program

        if self._mode == "1F1B":
            self._insert_sync_ops_for_1f1b()
            self._task_1f1b()
        elif self._mode == "F-Then-B":
            raise NotImplementedError("F-Then-B has not been implemented")
        elif self._mode == "stream":
            self._insert_sync_ops_for_stream()
            self._task_stream()
        else:
            raise ValueError(
                "Now only 'F-then-B', '1F1B' and 'stream' are supported."
                "The given value is {}.".format(self._mode)
            )

    def _insert_sync_ops_for_1f1b(self):
        """
        This implementation refers to lots of Paddle/python/paddle/fluid/optimizer.py.
        The difference between this function with 'PipelineOptimizer' is that
        'send_v2' op and 'recv_v2' op have been inserted in program by 'reshard'.
        """

        for block in self._program.blocks:
            offset = 0
            first_optimize_index = None
            for index, op in enumerate(list(block.ops)):
                if is_optimize_op(op):
                    first_optimize_index = index
                    break

            # insert sync ops
            for index, op in enumerate(list(block.ops)):
                if op.type == 'send_v2':
                    # step1: set 'use_calc_stream' False
                    op._set_attr("use_calc_stream", False)
                    op_role = op.attr('op_role')
                    ring_id = op.attr('ring_id')
                    # step2: insert 'c_sync_calc_stream' op before 'send_v2' op
                    var_name = op.input_arg_names[0]
                    var = block.var(var_name)
                    block._insert_op_without_sync(
                        index=index + offset,
                        type="c_sync_calc_stream",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={'op_role': op_role},
                    )
                    offset += 1
                    # step3: insert 'c_sync_comm_stream' op after 'send_v2' op or
                    # before the first optimize op
                    if int(op_role) == int(OpRole.Backward):
                        index = first_optimize_index + offset
                        new_op_role = OpRole.Optimize
                    else:
                        index = index + offset + 1
                        new_op_role = OpRole.Backward
                    sync_comm_op = block._insert_op_without_sync(
                        index=index,
                        type="c_sync_comm_stream",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={
                            'op_role': new_op_role,
                            'ring_id': ring_id,
                        },
                    )
                    # step4: If 'send_v2' op in forward parse, set 'pipeline_flag' to distinguish
                    # whether the 'c_sync_comm_stream' op is inserted for pipeline.
                    if int(op_role) == int(OpRole.Forward):
                        sync_comm_op._set_attr('pipeline_flag', '')
                        offset += 1
            block._sync_with_cpp()

            offset = 0
            backward_recv_index = None
            for index, op in enumerate(block.ops):
                if op.type == "recv_v2" and is_backward_op(op):
                    backward_recv_index = index
                    break
            if backward_recv_index is None:
                continue

            # replace 'c_sync_comm_stream' op with 'nop' op
            for index, op in enumerate(list(block.ops)):
                if index >= backward_recv_index:
                    break
                if op.type == 'c_sync_comm_stream' and op.has_attr(
                    'pipeline_flag'
                ):
                    var_name = op.output_arg_names[0]
                    var = block.var(var_name)
                    block._remove_op(index + offset, sync=False)
                    offset -= 1
                    block._insert_op_without_sync(
                        index=backward_recv_index,
                        type="nop",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={'op_role': OpRole.Backward},
                    )
            block._sync_with_cpp()

    def _insert_sync_ops_for_stream(self):

        for block in self._program.blocks:
            offset = 0
            send_vars = []
            # insert sync ops
            for index, op in enumerate(list(block.ops)):
                if op.type == 'send_v2':
                    # step1: set 'use_calc_stream' False
                    op._set_attr("use_calc_stream", False)
                    op_role = op.attr('op_role')
                    # step2: insert 'c_sync_calc_stream' op before 'send_v2' op
                    var_name = op.input_arg_names[0]
                    var = block.var(var_name)
                    block._insert_op_without_sync(
                        index=index + offset,
                        type="c_sync_calc_stream",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={'op_role': op_role},
                    )
                    offset += 1
                    send_vars.append(var_name)

            for var_name in send_vars:
                nop_op = block.append_op(type='nop')
                nop_op.desc.set_input('X', [var_name])
                nop_op.desc.set_output('Out', [var_name])

            block._sync_with_cpp()

    def _create_param(self, dst_block, src_var):
        copied_kwargs = {}
        copied_kwargs['trainable'] = src_var.trainable
        copied_kwargs['optimize_attr'] = src_var.optimize_attr
        copied_kwargs['regularizer'] = src_var.regularizer
        copied_kwargs['do_model_average'] = src_var.do_model_average
        copied_kwargs['need_clip'] = src_var.need_clip

        Parameter(
            block=dst_block,
            type=src_var.type,
            name=src_var.name,
            shape=src_var.shape,
            dtype=src_var.dtype,
            lod_level=src_var.lod_level,
            error_clip=src_var.error_clip,
            stop_gradient=src_var.stop_gradient,
            is_data=src_var.is_data,
            belong_to_optimizer=src_var.belong_to_optimizer,
            **copied_kwargs
        )

    def _create_inter(self, dst_block, src_var):
        dst_block.create_var(
            type=src_var.type,
            name=src_var.name,
            shape=src_var.shape,
            dtype=src_var.dtype,
            lod_level=src_var.lod_level,
            persistable=src_var.persistable,
            error_clip=src_var.error_clip,
            stop_gradient=src_var.stop_gradient,
            is_data=src_var.is_data,
            belong_to_optimizer=src_var.belong_to_optimizer,
        )

    def _create_var(
        self, src_block, dst_block, src_varname, force_create=False
    ):

        if not force_create:
            src_var = src_block.var(src_varname)
        else:
            src_var = src_block._var_recursive(src_varname)
        if src_var.type in __not_shape_var_type__:
            persist = getattr(src_var, 'persistable', False)
            dst_block.create_var(
                type=src_var.type,
                name=src_var.name,
                persistable=persist,
                error_clip=src_var.error_clip,
                stop_gradient=src_var.stop_gradient,
                is_data=src_var.is_data,
                belong_to_optimizer=src_var.belong_to_optimizer,
            )
        else:
            if isinstance(src_var, Parameter):
                self._create_param(dst_block, src_var)
            else:
                self._create_inter(dst_block, src_var)

    def _create_program(self, src_block, dst_block, src_op, force_create=False):
        dst_op_desc = dst_block.desc.append_op()
        dst_op_desc.copy_from(src_op.desc)
        for input_varname in src_op.input_arg_names:
            if src_block.has_var(input_varname) or (
                force_create and src_block._find_var_recursive(input_varname)
            ):
                self._create_var(
                    src_block, dst_block, input_varname, force_create
                )
        for output_varname in src_op.output_arg_names:
            if src_block.has_var(output_varname) or (
                force_create and src_block._find_var_recursive(output_varname)
            ):
                self._create_var(
                    src_block, dst_block, output_varname, force_create
                )

    def _get_pp_stage(self, rank):
        pp_idx = None
        for idx, process_mesh in enumerate(self._dist_context.process_meshes):
            if rank in process_mesh.processes:
                pp_idx = idx
                break
        return pp_idx

    def _task_stream(self):
        cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "").split(',')
        nrank = len(trainer_endpoints)
        num_of_functionality = 5

        # compute current pp stage
        pp_stages = len(self._dist_context.process_meshes)
        cur_pp_stage = self._get_pp_stage(cur_rank)

        print("pp_stages:", pp_stages)
        print("cur_rank:", cur_rank)
        print("cur_pp_stage:", cur_pp_stage)
        print("process_meshes:", self._dist_context.process_meshes)
        for process_mesh in self._dist_context.process_meshes:
            print("--> processes:", process_mesh.processes)

        start_prog = Program()
        cond_prog = Program()
        end_prog = Program()
        send_prog = Program()
        recv_prog = Program()

        cond_var_name = None
        for ib, src_block in enumerate(self._program.blocks):
            if ib == 0:
                strat_block = start_prog.block(0)
                end_block = end_prog.block(0)

                is_after_while_op = False
                for op in src_block.ops:
                    if op.type == "while":
                        assert len(op.input('Condition')) == 1
                        cond_var_name = op.input('Condition')[0]
                        is_after_while_op = True
                        continue

                    if not is_after_while_op:
                        self._create_program(
                            src_block, strat_block, op, force_create=True
                        )
                    else:
                        self._create_program(
                            src_block, end_block, op, force_create=True
                        )
            elif ib == 1:
                send_block = send_prog.block(0)
                recv_block = recv_prog.block(0)

                is_after_send_op = False
                is_after_recv_op = False
                for op in src_block.ops:
                    if op.type == "send_v2" and not is_after_send_op:
                        is_after_send_op = True
                        self._create_program(
                            src_block, send_block, op, force_create=True
                        )
                        continue

                    if (
                        is_after_send_op
                        and not is_after_recv_op
                        and op.type == "recv_v2"
                    ):
                        is_after_recv_op = True
                        self._create_program(
                            src_block, recv_block, op, force_create=True
                        )
                        continue

                    if not is_after_send_op or not is_after_recv_op:
                        self._create_program(
                            src_block, send_block, op, force_create=True
                        )

                    if is_after_send_op and is_after_recv_op:
                        self._create_program(
                            src_block, recv_block, op, force_create=True
                        )
            else:
                raise Exception("Only support generation condition.")

        start_prog._sync_with_cpp()
        end_prog._sync_with_cpp()
        send_prog._sync_with_cpp()
        recv_prog._sync_with_cpp()

        # print("=" * 20)
        # print("start_prog:")
        # print(start_prog)

        # print("=" * 20)
        # print("cond_prog:")
        # print(cond_prog)

        # print("=" * 20)
        # print("send_prog:")
        # print(send_prog)

        # print("=" * 20)
        # print("recv_prog:")
        # print(recv_prog)

        # print("=" * 20)
        # print("end_prog:")
        # print(end_prog)

        assert cond_var_name is not None

        start_task_node = TaskNode(
            rank=cur_rank,
            max_run_times=self._acc_steps,
            node_type="Start",
            task_id=int(cur_rank * num_of_functionality + 0),
            program=start_prog,
            lazy_initialize=True,
        )
        cond_task_node = TaskNode(
            rank=cur_rank,
            max_run_times=self._acc_steps,
            node_type="Cond",
            task_id=int(cur_rank * num_of_functionality + 1),
            program=cond_prog,
            cond_var_name=cond_var_name,
            lazy_initialize=True,
        )
        send_task_node = TaskNode(
            rank=cur_rank,
            max_run_times=self._acc_steps,
            node_type="Compute",
            task_id=int(cur_rank * num_of_functionality + 2),
            program=send_prog,
            lazy_initialize=True,
        )
        recv_task_node = TaskNode(
            rank=cur_rank,
            max_run_times=self._acc_steps,
            node_type="Compute",
            task_id=int(cur_rank * num_of_functionality + 3),
            program=recv_prog,
            lazy_initialize=True,
        )
        end_task_node = TaskNode(
            rank=cur_rank,
            max_run_times=self._acc_steps,
            node_type="Compute",
            task_id=int(cur_rank * num_of_functionality + 4),
            program=end_prog,
            lazy_initialize=True,
        )

        # add dependencies for task nodes intra stage
        inf = -1
        pp_buff_size = int(pp_stages - cur_pp_stage)
        start_task_node.add_downstream_task(
            cond_task_node.task_id(), self._gen_bsz
        )
        print(
            "Task ",
            start_task_node.task_id(),
            "'s downstream is:",
            cond_task_node.task_id(),
            ", buffer size is:",
            self._gen_bsz,
        )
        cond_task_node.add_upstream_task(
            start_task_node.task_id(), self._gen_bsz
        )
        print(
            "Task ",
            cond_task_node.task_id(),
            "'s upstream is:",
            start_task_node.task_id(),
            ", buffer size is:",
            self._gen_bsz,
        )
        cond_task_node.add_downstream_task(send_task_node.task_id(), inf)
        print(
            "Task ",
            cond_task_node.task_id(),
            "'s downstream is:",
            send_task_node.task_id(),
            ", buffer size is:",
            inf,
        )
        send_task_node.add_upstream_task(cond_task_node.task_id(), inf)
        print(
            "Task ",
            send_task_node.task_id(),
            "'s upstream is:",
            cond_task_node.task_id(),
            ", buffer size is:",
            inf,
        )
        send_task_node.add_downstream_task(
            recv_task_node.task_id(), pp_buff_size
        )
        print(
            "Task ",
            send_task_node.task_id(),
            "'s downstream is:",
            recv_task_node.task_id(),
            ", buffer size is:",
            pp_buff_size,
        )
        recv_task_node.add_upstream_task(send_task_node.task_id(), pp_buff_size)
        print(
            "Task ",
            recv_task_node.task_id(),
            "'s upstream is:",
            send_task_node.task_id(),
            ", buffer size is:",
            pp_buff_size,
        )
        recv_task_node.add_downstream_task(
            cond_task_node.task_id(), inf, core.DependType.LOOP
        )
        print(
            "Task ",
            recv_task_node.task_id(),
            "'s downstream is:",
            cond_task_node.task_id(),
            ", buffer size is:",
            inf,
        )
        cond_task_node.add_upstream_task(
            recv_task_node.task_id(), inf, core.DependType.LOOP
        )
        print(
            "Task ",
            cond_task_node.task_id(),
            "'s upstream is:",
            recv_task_node.task_id(),
            ", buffer size is:",
            inf,
        )
        cond_task_node.add_downstream_task(
            end_task_node.task_id(), inf, core.DependType.STOP_LOOP
        )
        print(
            "Task ",
            cond_task_node.task_id(),
            "'s downstream is:",
            end_task_node.task_id(),
            ", buffer size is:",
            inf,
        )
        end_task_node.add_upstream_task(
            cond_task_node.task_id(), inf, core.DependType.STOP_LOOP
        )
        print(
            "Task ",
            end_task_node.task_id(),
            "'s upstream is:",
            cond_task_node.task_id(),
            ", buffer size is:",
            inf,
        )

        # add dependencies for task nodes inter stage
        # get upstream ranks and downstream ranks of cur_rank
        up_down_streams = self._dist_context.up_down_streams
        pp_upstream_ranks = up_down_streams.ups(cur_rank)
        pp_downstream_ranks = up_down_streams.downs(cur_rank)

        for upstream_rank in pp_upstream_ranks:
            upstream_pp_stage = self._get_pp_stage(upstream_rank)
            if upstream_pp_stage < pp_stages - 1:
                upstream_task_id = int(upstream_rank * num_of_functionality + 2)
                send_task_node.add_upstream_task(upstream_task_id)
                print(
                    "Task ",
                    send_task_node.task_id(),
                    "'s upstream is:",
                    upstream_task_id,
                    ", buffer size is:",
                    2,
                )
            else:
                upstream_task_id = int(upstream_rank * num_of_functionality + 3)
                recv_task_node.add_upstream_task(upstream_task_id)
                print(
                    "Task ",
                    recv_task_node.task_id(),
                    "'s upstream is:",
                    upstream_task_id,
                    ", buffer size is:",
                    2,
                )
        for downstream_rank in pp_downstream_ranks:
            if cur_pp_stage < pp_stages - 1:
                downstream_task_id = int(
                    downstream_rank * num_of_functionality + 2
                )
                send_task_node.add_downstream_task(downstream_task_id)
                print(
                    "Task ",
                    send_task_node.task_id(),
                    "'s downstream is:",
                    downstream_task_id,
                    ", buffer size is:",
                    2,
                )
            else:
                downstream_task_id = int(
                    downstream_rank * num_of_functionality + 3
                )
                recv_task_node.add_downstream_task(downstream_task_id)
                print(
                    "Task ",
                    recv_task_node.task_id(),
                    "'s downstream is:",
                    downstream_task_id,
                    ", buffer size is:",
                    2,
                )

        task_id_to_rank = {}
        for i in range(nrank):
            for j in range(num_of_functionality):
                task_id_to_rank[int(i * num_of_functionality + j)] = i
        self._program._pipeline_opt = {
            "fleet_opt": {
                'tasks': [
                    start_task_node,
                    cond_task_node,
                    send_task_node,
                    recv_task_node,
                    end_task_node,
                ],
                'task_id_to_rank': task_id_to_rank,
                'num_micro_batches': self._acc_steps,
                'inference_generation': True,
            }
        }
