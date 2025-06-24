import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPadding(nn.Module):
	def __init__(self,padding_value):
		super(CustomPadding, self).__init__()
		self.padding_value  = padding_value

	def forward(self, tokens_batch):
		# 使用 pad_sequence 对批量中的每个序列进行填充
		padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(
			s) for s in tokens_batch], batch_first=True, padding_value=self.padding_value)

		# 创建 mask，非 padding 部分为 False， padding 部分为 True
		padding_mask = padded_batch == self.padding_value  # padding 部分是 True

		return padded_batch, padding_mask


class CustomTransformerBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
		super(CustomTransformerBlock, self).__init__()

		# 多头注意力层
		self.mha = nn.MultiheadAttention(
			embed_dim, num_heads, dropout=dropout, batch_first=True)

		# 前馈神经网络 (FFN)
		self.ffn = nn.Sequential(
			nn.Linear(embed_dim, ffn_hidden_dim),  # 第一层
			nn.ReLU(),  # 激活函数
			nn.Linear(ffn_hidden_dim, ffn_hidden_dim),  # 第 2 层
			nn.ReLU(), 
			nn.Linear(ffn_hidden_dim, embed_dim)  # 回到原始维度
		)

		# 残差连接和 LayerNorm
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, keypadding_mask):
		# 1️⃣ 多头注意力
		attn_output, _ = self.mha(
			x, x, x, key_padding_mask=keypadding_mask)  # Self-Attention
		x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + 归一化

		# 2️⃣ 前馈神经网络 (FFN)
		ffn_output = self.ffn(x)
		x = self.norm2(x + self.dropout(ffn_output))  # 残差连接 + 归一化

		return x


class CustomTransformerClassifier(nn.Module):
	def __init__(self, vocab_size, max_seq_len,  embed_dim, num_heads, ffn_hidden_dim, num_classes,  dropout=0.1):
		super(CustomTransformerClassifier, self).__init__()

		# Step 1: Token padding
		self.padding = CustomPadding(padding_value=num_classes)

		# Step 1: Token Embedding
		self.embedding = nn.Embedding(vocab_size, embed_dim)

		# Step 2: Positional Encoding
		self.positional_encoding = nn.Parameter(
			torch.zeros(1, max_seq_len, embed_dim))

		# # Step 3: 多头注意力 + FFN 层
		self.transformer_block = CustomTransformerBlock(
			embed_dim, num_heads, ffn_hidden_dim, dropout)

		# Step 4: 分类头（最终输出 K 维 logit）
		self.classifier = nn.Linear(embed_dim, num_classes)

	def forward(self, x):
		actionIndexDel = [ seq[-1] for seq in x]
		actionIndexDel = torch.tensor(actionIndexDel)
		# Step 1：Token padding
		x, keypadding_mask = self.padding(x)

		# Step 1: Token Embedding
		x = self.embedding(x)  # (batch_size, seq_len, embed_dim)

		# Step 2: Add Positional Encoding
		seq_len = x.size(1)
		# (batch_size, seq_len, embed_dim)
		x = x + self.positional_encoding[:, :seq_len, :]

		x = self.transformer_block(x, keypadding_mask)  # Transformer 处理

		# 平均池化 (也可以使用 [CLS] Token)
		# (batch_size, seq_len, 1)
		mask = (~keypadding_mask).unsqueeze(-1).float()
		x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, embed_dim)

		logits = self.classifier(x)  # (batch_size, num_classes)
		logits = logits.scatter(1, actionIndexDel.unsqueeze(1), float('-inf'))
		return logits


if __name__ == '__main__':

# # 示例：测试padding
# 	tokens_batch = [
# 		[0, 2, 3],       # 第一句
# 		[4, 5],          # 第二句
# 		[1, 7, 6, 8, 0]     # 第三句
# 	]

# 	actionIndexDel = [ seq[-1] for seq in tokens_batch]
# 	print("\nactionDel:", actionIndexDel)

# # 示例：测试padding
# # 创建 Padding 类实例
# 	padding_value=9
# 	padding = CustomPadding(padding_value)

# # 对 tokens_batch 进行 padding 和生成 mask
	
# 	padded_batch, padding_mask = padding(tokens_batch )

# 	print("Padded Batch:")
# 	print(padded_batch)
# 	print("\nPadding Mask:")
# 	print(padding_mask)

# # 示例： 测试embedding
# 	vocab_size = 10  # 需要大于token里面的最大数， token的取值范围为[0, vocab_size-1]
# 	embed_dim = 2
# 	embedding = nn.Embedding(vocab_size, embed_dim)
# 	embed_batch = embedding(padded_batch)
# 	print("\n embed Batch:")
# 	print(embed_batch)

# # 示例：测试positional embedding
# 	max_seq_len = 20
# 	positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
# 	seq_len = embed_batch.size(1)
# 	embed_batch = embed_batch + positional_encoding[:, :seq_len, :]
# 	print("\n position embed Batch:", embed_batch.shape)
# 	print(embed_batch)

# # 示例：测试 self-attention transformer
# 	num_heads = 1
# 	ffn_hidden_dim = embed_dim*10
# 	dropout = 0.1
# 	transformer_block = CustomTransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)
# 	x = transformer_block(x=embed_batch, keypadding_mask=padding_mask)
# 	print("\n attention x :", x.shape)
# 	print(x)

# # 示例：测试 分类器
# 	num_classes = vocab_size-1
# 	classifier = nn.Linear(embed_dim, num_classes)
# 	mask = (~padding_mask).unsqueeze(-1).float()
# 	x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, embed_dim)
# 	print("\n 池化后 x :", x.shape)
# 	print(x)
# 	logits = classifier(x)  # (batch_size, num_classes)
# 	print("\n logits :", logits.shape)
# 	print(logits)
# 	print("probabilities:", F.softmax(logits, dim=-1),
# 	  torch.sum(F.softmax(logits, dim=-1), dim=-1))


# # 示例： 测试 实例化模型
# 	TransformerClassifier = CustomTransformerClassifier(vocab_size, max_seq_len, embed_dim, num_heads, ffn_hidden_dim,num_classes)
# 	logits = TransformerClassifier(tokens_batch)  # (batch_size, num_classes)
# 	print("\n logits :", logits.shape)
# 	print(logits)
# 	print("probabilities:", F.softmax(logits, dim=-1),torch.sum(F.softmax(logits, dim=-1), dim=-1))


	Nq = 3
	Depth = 10
	embed_dim = 10
	num_heads = 1
	TransformerClassifier = CustomTransformerClassifier(vocab_size = Nq*Nq+1, max_seq_len=Depth, embed_dim= embed_dim, num_heads=num_heads, ffn_hidden_dim=embed_dim*5, num_classes= Nq*Nq)

	num_classes= Nq*Nq
	vocab_size = Nq*Nq+1
	batch_size = 5
	tokens_batch = [ torch.randint(0,num_classes, (i+1,) ).tolist() for i in range(batch_size)]
	print("\ntokens_batch", len(tokens_batch) )
	print(tokens_batch)
	logits = TransformerClassifier(tokens_batch)  # (batch_size, num_classes)
	print("\nlogits :", logits.shape)
	print(logits)
	print("probabilities:", F.softmax(logits, dim=-1),torch.sum(F.softmax(logits, dim=-1), dim=-1))