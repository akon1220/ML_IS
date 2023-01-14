# data_loading_utils.py: a file defining general classes and functions for loading (esp. for training) in the cross-domain-learning repository
# SEE LICENSE STATEMENT AT THE END OF THE FILE

# dependency import statements

def extract_necessary_elements_from_layerwise_sample(samp, samp_type, requested_tasks):
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: samp == ", samp)
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: samp_type == ", samp_type)
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: requested_tasks == ", requested_tasks)
    samp = list(samp)
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: len(samp) == ", len(samp))
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: samp[0].size() == ", samp[0].size())

    # save anchor sample
    assert samp_type in ["AnchoredBTUABRPTS", "NonAnchoredBTUABRPTS"]
    necessary_elements = [samp[0]] # assumes samp_type in ["AnchoredBTUABRPTS", "NonAnchoredBTUABRPTS"]

    samp_inds_to_keep = []
    for task_id in requested_tasks:
        if samp_type == "AnchoredBTUABRPTS": # an anchor, rp other, ts anchor2, and ts other window along with label
            if task_id == 'RP':
                samp_inds_to_keep.append(1)
            elif task_id == 'TS':
                samp_inds_to_keep = samp_inds_to_keep + [2,3]
            elif 'Behavioral' in task_id:
                pass # this case is handled at necessary_elements initialization
            else:
                raise ValueError("data_loading_utils.extract_necessary_elements_from_layerwise_sample: Unrecognized task_id == "+str(task_id))
        elif samp_type == "NonAnchoredBTUABRPTS": # a behavioral sample window, 2 rp windows, and 3 ts windows
            if task_id == 'RP':
                samp_inds_to_keep = samp_inds_to_keep + [1,2]
            elif task_id == 'TS':
                samp_inds_to_keep = samp_inds_to_keep + [3,4,5]
            elif 'Behavioral' in task_id:
                pass # this case is handled at necessary_elements initialization
            else:
                raise ValueError("data_loading_utils.extract_necessary_elements_from_layerwise_sample: Unrecognized task_id == "+str(task_id))
        else:
            raise ValueError("data_loading_utils.extract_necessary_elements_from_layerwise_sample: Unrecognized sample type == "+str(samp_type))
    
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: samp_inds_to_keep == ", samp_inds_to_keep)
    necessary_elements = necessary_elements + [samp[x] for x in sorted(samp_inds_to_keep)] + [samp[y] for y in [-3,-2,-1]] # don't forget the labels at samp[-3:] # + [samp[-1]]
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: len(necessary_elements) == ", len(necessary_elements))
    print("data_loading_utils.extract_necessary_elements_from_layerwise_sample: necessary_elements[0].size() == ", necessary_elements[0].size())
    return necessary_elements
