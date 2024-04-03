import torch


class GlobalNormalization1(torch.nn.Module):
    def __init__(self, feature_dim, scale=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("running_ave", torch.zeros(1, self.feature_dim, 1))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        if self.scale:
            self.register_buffer("running_sq_diff", torch.zeros(1, self.feature_dim, 1))


    def forward(self, inputs):
        if self.training:
            # Update running estimates of statistics
            frames_in_input = inputs.shape[0] * inputs.shape[2]
            updated_running_ave = (
                self.running_ave * self.total_frames_seen + inputs.sum(dim=(0, 2), keepdim=True)
            ) / (self.total_frames_seen + frames_in_input)
            if self.scale:
                # Update the sum of the squared differences between inputs and mean
                self.running_sq_diff = self.running_sq_diff + (
                    (inputs - self.running_ave) * (inputs - updated_running_ave)
                ).sum(dim=(0, 2), keepdim=True)
            self.running_ave = updated_running_ave
            self.total_frames_seen = self.total_frames_seen + frames_in_input
        else:
            return inputs

        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = (inputs - self.running_ave) / std
        else:
            inputs = inputs - self.running_ave

        return inputs
    
    def unnorm(self, inputs):

        if self.scale:

            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)

            inputs = inputs*std + self.running_ave

        else:

            inputs = inputs + self.running_ave



        return inputs
