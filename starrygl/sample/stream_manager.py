import torch
class WorkStreamEvent:
    def __init__(self):
        self.train_stream = torch.cuda.Stream()
        self.write_memory_stream = torch.cuda.Stream()
        self.fetch_stream = torch.cuda.Stream()
        self.write_mail_stream = torch.cuda.Stream()
        self.event = None
event = WorkStreamEvent()
def get_event():
    return event.event
