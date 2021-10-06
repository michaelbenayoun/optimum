#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from torch.utils.data import DataLoader


class LpotDataLoader(DataLoader):

    @classmethod
    def from_pytorch_dataloader(cls, dataloader: DataLoader):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected a PyTorch DataLoader, got: {type(dataloader)}.")
        lpot_dataloader = cls(dataloader.dataset)
        for key, value in dataloader.__dict__.items():
            lpot_dataloader.__dict__[key] = value
        return lpot_dataloader

    def __iter__(self):
        for input in super().__iter__():
            if not isinstance(input, (dict, tuple, list)):
                raise TypeError(f"Model calibration cannot use input of type {type(input)}.")
            label = input.get("labels") if isinstance(input, dict) else None
            yield input, label


def _cfgs_to_fx_cfgs(op_cfgs, observer_type='post_training_static_quant'):
    """LPOT function which convert a quantization config to a format that meets the requirements of torch.fx.
        Args:
            op_cfgs (dict): dictionary of quantization configure for each op
            observer_type (str, optional): specify observer type, Default is 'ptq_static',
                                           options: 'ptq_dynamic', 'qat'.
        Returns:
            fx_op_cfgs (dict): dictionary of quantization configure that meets the requirements of torch.fx
    """
    fx_op_cfgs = dict()
    op_tuple_cfg_list = []
    for key, value in op_cfgs.items():
        if key == "default_qconfig":
            fx_op_cfgs[""] = value
            continue
        op_tuple = (key, value)
        op_tuple_cfg_list.append(op_tuple)
    fx_op_cfgs["module_name"] = op_tuple_cfg_list
    return fx_op_cfgs


def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
    """LPOT helper function for `query_fw_capability` which get all quantizable ops from model.
    Args:
        model (object): input model
        prefix (string): prefix of op name
        quantizable_ops (list): list of quantizable ops from model include op name and type.
    Returns:
        None
    """
    import torch
    from lpot.adaptor.pytorch import unify_op_type_mapping

    for name, child in model.named_children():
        op_name = prefix + '.' + name if prefix != '' else name
        if type(child) in self.white_list and type(child) != torch.nn.Sequential and \
                type(child) != torch.quantization.stubs.DeQuantStub and not \
                    isinstance(child, torch.nn.LayerNorm) and not \
                    isinstance(child, torch.nn.Embedding):

            quantizable_ops.append((
                op_name, unify_op_type_mapping[str(child.__class__.__name__)]
                if str(child.__class__.__name__) in unify_op_type_mapping else
                str(child.__class__.__name__)))
        else:
            self._get_quantizable_ops_recursively(child, op_name, quantizable_ops)

