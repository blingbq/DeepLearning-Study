import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,heads,d_model,dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.d_k=d_model//heads
        self.h=heads

        self.q_linear=nn.Linear(d_model,d_model)
        self.k_linear=nn.Linear(d_model,d_model)
        self.v_linear=nn.Linear(d_model,d_model)

        self.dropout=nn.Dropout(dropout)

        self.out=nn.Linear(d_model,d_model)

    def attention(self,q,k,v,mask=None):
        scores=torch.matmul(q,k.transpose(-2,-1)/math.sqrt(self.d_k))
        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)
        
        scores=F.softmax(scores,dim=-1)

        scores=self.dropout(scores)

        output=torch.matmul(scores,v)

        return output
    
    def forward(self,q,k,v,mask=None):
        batch_size=q.size(0)

        q=self.q_linear(q).view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        k=self.k_linear(k).view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        v=self.v_linear(v).view(batch_size,-1,self.h,self.d_k).transpose(1,2)

        scores = self.attention(q,k,v,mask)

        concat=scores.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        output=self.out(concat)

        return output
    
if __name__=="__main__":
    heads=4
    d_model=128
    dropout=0.1
    model=MultiHeadAttention(heads,d_model,dropout)

    batch_size=2
    seq_len=5

    q=torch.randn(batch_size,seq_len,d_model)
    k=torch.randn(batch_size,seq_len,d_model)
    v=torch.randn(batch_size,seq_len,d_model)

    output=model(q,k,v)
    print("Output shape:", output.shape)  # 应该是 (batch_size, seq_len, d_model)
    loss=output.mean()  # 示例损失计算
    loss.backward()  # 反向传播示例
    print("Loss:", loss.item())  # 打印损失值
    print("Backward pass completed successfully.")
