from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, matrix1, matrix2):
        matrix1_flat = matrix1.view(-1)
        matrix2_flat = matrix2.view(-1)

        matrix_connect = torch.cat((matrix1_flat, matrix2_flat), dim=0)

        output = self.fc(matrix_connect)
        output_matrix = output.view(4, 4)
        return output_matrix