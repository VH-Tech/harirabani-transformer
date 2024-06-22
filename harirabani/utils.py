import torch
# from matplotlib import pyplot as plt
from prettytable import PrettyTable



def count_parameters_trainable(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        summery = table.get_string()
    with open("model_summery.txt", "w") as f:
        f.write(summery)
        f.write("\n Total Trainable Params:")
        f.write(str(total_params))
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    return total_params


def count_parameters_total(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        summery = table.get_string()
    with open("model_summary.txt", "w") as f:
        f.write(summery)
        f.write("\n Total Trainable Params:")
        f.write(str(total_params))
    # print(table)
    # print(f"Total Params: {total_params}")
    return total_params



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):

    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count