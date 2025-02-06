import torch
import torch.nn as nn
import torch.nn.functional as F


# * 最基础的 Mixture of Experts 模型
class BasicExpert(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class BasicMoE(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList(
            [BasicExpert(in_dim, out_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        gates = torch.softmax(self.gate(x), dim=-1)
        expert_outs = torch.cat(
            [expert(x).unsqueeze(1) for expert in self.experts], dim=1
        )
        return torch.sum(gates.unsqueeze(-1) * expert_outs, dim=1)


# * Sparse Mixture of Experts
class MoERouter(nn.Module):
    """稀疏混合专家路由模块"""

    def __init__(self, hidden_dim, num_experts, top_k=3, eps=1e-6):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.eps = eps  # 防止除以零的小量

    def forward(self, x):
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # 选取 top-k 专家
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # 归一化权重（数值稳定性优化）
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + self.eps)

        # 生成专家掩码（one-hot 编码）
        expert_mask = F.one_hot(top_k_indices, self.num_experts).sum(
            dim=1
        )  # (batch*seq, num_experts)

        return router_logits, top_k_weights, top_k_indices, expert_mask.float()


class SparseMoE(nn.Module):
    """稀疏混合专家模块"""

    def __init__(self, hidden_dim, num_experts, top_k=4, expert_class=BasicExpert):
        super().__init__()
        self.router = MoERouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            [expert_class(hidden_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # (batch*seq, hidden_dim)

        # 路由计算
        router_logits, top_k_weights, top_k_indices, expert_mask = self.router(x_flat)

        # 初始化输出张量
        moe_output = torch.zeros_like(x_flat)

        # 遍历每个 top-k 专家
        for i in range(self.top_k):
            # 获取第 i 个专家对应的索引和权重
            expert_idx = top_k_indices[:, i]  # (batch*seq,)
            weight = top_k_weights[:, i].unsqueeze(1)  # (batch*seq, 1)

            # 选择对应的专家输出
            selected_input = torch.zeros_like(x_flat)
            for idx, expert in enumerate(self.experts):
                mask = (expert_idx == idx).float().unsqueeze(1)  # (batch*seq, 1)
                selected_input += mask * expert(x_flat)

            # 加权累加
            moe_output += weight * selected_input

        # 恢复原始形状
        moe_output = moe_output.view(batch_size, seq_len, hidden_dim)

        # 计算负载均衡损失
        aux_loss = self._load_balancing_loss(expert_mask, router_logits)

        return moe_output, aux_loss

    def _load_balancing_loss(self, expert_mask, router_logits):
        """负载均衡损失（防止某些专家被过度使用）"""
        # 计算每个专家被选中的概率
        expert_gates = F.softmax(router_logits, dim=-1)  # (batch * seq, num_experts)
        expert_usage = expert_mask.mean(dim=0)  # (num_experts,)
        gate_usage = expert_gates.mean(dim=0)  # (num_experts,)

        # 负载均衡损失系数（可调整）
        return torch.dot(expert_usage, gate_usage) * self.num_experts


class SharedMoE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_shared_experts,
        num_sparse_experts,
        expert_class=BasicExpert,
    ):
        super().__init__()
        self.sparse_experts = SparseMoE(hidden_dim, num_sparse_experts)
        self.shared_experts = nn.ModuleList(
            [expert_class(hidden_dim, hidden_dim) for _ in range(num_shared_experts)]
        )

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # (batch*seq, hidden_dim)

        # 专家路由计算
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 初始化输出张量
        moe_output = torch.zeros_like(x_flat)

        # 遍历每个专家
        for i, expert in enumerate(self.experts):
            expert_output = expert(x_flat)
            moe_output += gate_probs[:, i].unsqueeze(1) * expert_output

        # 恢复原始形状
        moe_output = moe_output.view(batch_size, seq_len, hidden_dim)

        return moe_output, gate_logits


if __name__ == "__main__":
    basic_moe = BasicMoE(10, 5, 16)
    x = torch.randn(4, 10)
    print(basic_moe(x).shape)

    sparse_moe = SparseMoE(512, 16)
    x = torch.randn(4, 10, 512)
    print(sparse_moe(x).shape)
