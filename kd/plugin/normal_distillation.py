import torch
from contextlib import ExitStack

def distillation(teacher, student, train_dataloader, recorders, pipe_rec, losses, optim_s, sched_s=None, device='cpu'):
    student.train()
    for _, value in pipe_rec.items():
        value['pipe'].train()
    teacher.eval()
    loss_batch = 0
    for data, label in train_dataloader:
        optim_s.zero_grad()
        data, label = data.to(device), label.to(device)
        with ExitStack() as stack:
            [stack.enter_context(value) for _, value in recorders.items()]
            with torch.no_grad():
                teacher(data)
            student(data)
            loss_total = 0
            for _, value in losses.items():
                params_list = []
                for pipe_name in value['input']:
                    if pipe_name == 'ground_truth':
                        params_list.append(label)
                    else:
                        recorder_name = pipe_rec[pipe_name]['rec_feature']
                        feature = recorders[recorder_name].get_record_data()
                        params_list.append(pipe_rec[pipe_name]['pipe'](feature))
                loss = value['weight'] * value['loss_fn'](*params_list)
                loss_total += loss
        loss_batch += loss_total.item()
        loss_total.backward()
        optim_s.step()
    if sched_s:
        sched_s.step()
    loss_batch /= len(train_dataloader)
    return loss_batch
