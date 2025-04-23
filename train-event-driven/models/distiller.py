import torch
import torch.nn as nn

#### Distiller
class BaseDistiller(nn.Module) :

    def __init__(self, student:nn.Module, teacher:nn.Module, student_dist_mods=[], teacher_dist_mods=None) :
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.student_dist_mods = student_dist_mods
        self.teacher_dist_mods = teacher_dist_mods
        if teacher_dist_mods == None :
            self.teacher_dist_mods = student_dist_mods
        self.teacher_fea = []
        self.student_fea = []

        for name, m in self.student.named_modules() :
            if name in self.student_dist_mods :
                m.register_forward_hook(self.student_feature_hook)
        
        for name, m in self.teacher.named_modules() :
            if name in self.teacher_dist_mods :
                m.register_forward_hook(self.teacher_feature_hook)

    def teacher_feature_hook(self, module, fea_in, fea_out) :
        self.teacher_fea.append(fea_out.detach())
    
    def student_feature_hook(self, module, fea_in, fea_out) :
        self.student_fea.append(fea_out)
    
    def parameters(self) :
        return self.student.parameters()

    def forward(self, x, criterion) :

        self.teacher_fea.clear()
        self.student_fea.clear()

        teacher_out = self.teacher(x).detach()
        student_out = self.student(x)
        assert(len(self.teacher_fea) == len(self.student_fea))

        loss = 0

        for i in range(len(self.teacher_fea)) :
            loss += criterion(self.student_fea[i], self.teacher_fea[i])
        
        return student_out, loss
        # return student_out, 0