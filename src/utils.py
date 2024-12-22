import torch
import nnsight

from tqdm.auto import tqdm
from nnsight import LanguageModel
from typing import List, Tuple, Union

def path_patch(
    lm : LanguageModel, 
    root_components : List[Tuple[str, Union[int, List[int]]]], 
    path_components : List[str],
    src : str, 
    target : str
):  
  component_dict = {k: v for (k, v) in lm.model.named_modules() if any([name in k for name in ["norm", "self_attn", "mlp"]])}

  for component, position in root_components:
    assert component in component_dict, f"{component} is not a valid model component"

  for component in path_components:
    assert component in component_dict, f"{component} is not a valid model component"

  path_list = path_components + [comp for comp, pos in root_components]

  with torch.no_grad(): 
    # Get the activations of the components we're going to patch
    with lm.trace(src) as tracer:
      src_acts = {(comp, pos): component_dict[comp].output[:, pos, :].save() for comp, pos in root_components} 

    # Get the activations of the components we're not going to let be affected
    with lm.trace(target) as tracer:
      plain_acts = {comp: component_dict[comp].output.save() for comp in component_dict if comp not in path_list}

    # Patch in and save the results
    with lm.trace(target) as tracer:
      for comp in plain_acts:
        component_dict[comp].output = plain_acts[comp]
      for comp, pos in src_acts:
        component_dict[comp].output[:, pos, :] = src_acts[(comp, pos)]

      logits = lm.lm_head.output.save()
    return logits

def plain_run(lm : LanguageModel, prompt : str):
  with lm.trace(prompt) as tracer:
    return lm.lm_head.output.save()