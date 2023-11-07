---
lang: en
---
# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
[https://ar5iv.labs.arxiv.org/html/2205.14135](https://ar5iv.labs.arxiv.org/html/2205.14135)

---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

Tri Dao Department of Computer Science, Stanford University Daniel Y. Fu Department of Computer Science, Stanford University Stefano Ermon Department of
Computer Science, Stanford University Atri Rudra Department of Computer Science and Engineering, University at Buffalo, SUNY Christopher Ré Department of
Computer Science, Stanford University

###### Abstract

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate
attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock
speedup. We argue that a missing principle is making attention algorithms IO-aware---accounting for reads and writes between levels of GPU memory. We propose
FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and
GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a
range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing
approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length
512) compared to the MLPerf 1.1 training speed record, 3$\times$ speedup on GPT-2 (seq. length 1K), and 2.4$\times$ speedup on long-range arena (seq. length
1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and
6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the
Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

## 1 Introduction

Transformer models \[[82](#bib.bib82){.ltx_ref}\] have emerged as the most widely used architecture in applications such as natural language processing and
image classification. Transformers have grown larger \[[5](#bib.bib5){.ltx_ref}\] and deeper \[[83](#bib.bib83){.ltx_ref}\], but equipping them with longer
context remains difficult \[[80](#bib.bib80){.ltx_ref}\], since the self-attention module at their heart has time and memory complexity quadratic in sequence
length. An important question is whether making attention faster and more memory-efficient can help Transformer models address their runtime and memory
challenges for long sequences.

Many approximate attention methods have aimed to reduce the compute and memory requirements of attention. These methods range from
sparse-approximation \[[51](#bib.bib51){.ltx_ref}, [74](#bib.bib74){.ltx_ref}\] to low-rank approximation \[[84](#bib.bib84){.ltx_ref},
[50](#bib.bib50){.ltx_ref}, [12](#bib.bib12){.ltx_ref}\], and their combinations \[[3](#bib.bib3){.ltx_ref}, [92](#bib.bib92){.ltx_ref},
[9](#bib.bib9){.ltx_ref}\]. Although these methods reduce the compute requirements to linear or near-linear in sequence length, many of them do not display
wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate
with wall-clock speed) and tend to ignore overheads from memory access (IO).

![Refer to caption](/html/2205.14135/assets/x1.png){#S1.F1.g1 .ltx_graphics .ltx_centering .ltx_img_landscape width="761" height="297"}

Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In
the outer loop (red arrows), FlashAttention loops through blocks of the $\mathbf{K}$ and $\mathbf{V}$ matrices and loads them to fast on-chip SRAM. In each
block, FlashAttention loops over blocks of $\mathbf{Q}$ matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to
HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large $N \times N$ attention matrix to
HBM, resulting in an 7.6$\times$ speedup on the attention computation.

In this paper, we argue that a missing principle is making attention algorithms IO-aware \[[1](#bib.bib1){.ltx_ref}\]---that is, carefully accounting for reads
and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or
HBM \[[45](#bib.bib45){.ltx_ref}\],
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref} left). On modern GPUs,
compute speed has out-paced memory speed \[[61](#bib.bib61){.ltx_ref}, [62](#bib.bib62){.ltx_ref}, [63](#bib.bib63){.ltx_ref}\], and most operations in
Transformers are bottlenecked by memory accesses \[[43](#bib.bib43){.ltx_ref}\]. IO-aware algorithms have been critical for similar memory-bound operations,
when reading and writing data can account for a large portion of the runtime---such as database joins \[[71](#bib.bib71){.ltx_ref}\], image
processing \[[70](#bib.bib70){.ltx_ref}\], numerical linear algebra \[[4](#bib.bib4){.ltx_ref}\], and more \[[85](#bib.bib85){.ltx_ref},
[40](#bib.bib40){.ltx_ref}\]. However, common Python interfaces to deep learning such as PyTorch and Tensorflow do not allow fine-grained control of memory
access.

We propose FlashAttention, a new attention algorithm that computes exact attention with far fewer memory accesses. Our main goal is to avoid reading and writing
the attention matrix to and from HBM. This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large
intermediate attention matrix for the backward pass. We apply two well-established techniques to address these challenges. (i) We restructure the attention
computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction (also known as
tiling). (ii) We store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the backward pass, which is faster than
the standard approach of reading the intermediate attention matrix from HBM. We implement FlashAttention in CUDA to achieve fine-grained control over memory
access and fuse all the attention operations into one GPU kernel. Even with the increased FLOPs due to recomputation, our algorithm both runs faster (up to 7.6x
on GPT-2 \[[67](#bib.bib67){.ltx_ref}\],
Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref} right) and uses less
memory---linear in sequence length---than standard attention, thanks to the massively reduced amount of HBM access.

We analyze the IO complexity \[[1](#bib.bib1){.ltx_ref}\] of FlashAttention, proving that it requires
$O\operatorname{}{({N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}})}$ HBM accesses where $d$ is the head dimension and $M$ is the size of SRAM, as compared to
$\Omega\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ of standard attention. For typical values of $d$ and $M$, FlashAttention requires many times fewer HBM
accesses compared to standard attention (up to 9$\times$ fewer, as shown
in [Fig. 2](#S3.F2 "Figure 2 ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
Moreover, we provide a lower bound, showing that no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes.

We also show that FlashAttention can serve as a useful primitive for realizing the potential of approximate attention algorithms by overcoming their issues with
memory access overhead. As a proof of concept, we implement block-sparse FlashAttention, a sparse attention algorithm that is 2-4$\times$ faster than even
FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than FlashAttention by a factor
proportional to the sparsity ratio. We discuss further extensions to other operations (attention on multi-GPU, kernel regression, block-sparse matrix multiply)
in [Section 5](#S5 "5 Limitations and Future Directions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}. We
open-source FlashAttention to make it easier to build on this primitive.^1^^1^1FlashAttention code is available at
[https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention){.ltx_ref .ltx_url .ltx_font_typewriter target="_blank"}

We empirically validate that FlashAttention speeds up model training and improves model quality by modeling longer context. We also benchmark the runtime and
memory footprint of FlashAttention and block-sparse FlashAttention compared to prior attention implementations.

-   •
    Faster Model Training. FlashAttention trains Transformer models faster in wall-clock time. We train BERT-large (seq. length 512) 15% faster than the
    training speed record in MLPerf 1.1 \[[58](#bib.bib58){.ltx_ref}\], GPT2 (seq. length 1K) 3$\times$ faster than baseline implementations from
    HuggingFace \[[87](#bib.bib87){.ltx_ref}\] and Megatron-LM \[[77](#bib.bib77){.ltx_ref}\], and long-range arena (seq. length 1K-4K) 2.4$\times$ faster than
    baselines.
-   •
    Higher Quality Models. FlashAttention scales Transformers to longer sequences, which improves their quality and enables new capabilities. We observe a 0.7
    improvement in perplexity on GPT-2 and 6.4 points of lift from modeling longer sequences on long-document classification \[[13](#bib.bib13){.ltx_ref}\].
    FlashAttention enables the first Transformer that can achieve better-than-chance performance on the Path-X \[[80](#bib.bib80){.ltx_ref}\] challenge, solely
    from using a longer sequence length (16K). Block-sparse FlashAttention enables a Transformer to scale to even longer sequences (64K), resulting in the first
    model that can achieve better-than-chance performance on Path-256.
-   •
    Benchmarking Attention. FlashAttention is up to 3$\times$ faster than the standard attention implementation across common sequence lengths from 128 to 2K
    and scales up to 64K. Up to sequence length of 512, FlashAttention is both faster and more memory-efficient than any existing attention method, whereas for
    sequence length beyond 1K, some approximate attention methods (e.g., Linformer) start to become faster. On the other hand, block-sparse FlashAttention is
    faster than all existing approximate attention methods that we know of.

## 2 Background

We provide some background on the performance characteristics of common deep learning operations on modern hardware (GPUs). We also describe the standard
implementation of attention.

### 2.1 Hardware Performance

We focus here on GPUs. Performance on other hardware accelerators are similar \[[48](#bib.bib48){.ltx_ref}, [46](#bib.bib46){.ltx_ref}\].

GPU Memory Hierarchy. The GPU memory hierarchy
([Fig. 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref} left) comprises multiple
forms of memory of different sizes and speeds, with smaller memory being faster. As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with
bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s \[[45](#bib.bib45){.ltx_ref},
[44](#bib.bib44){.ltx_ref}\]. The on-chip SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size. As compute has gotten
faster relative to memory speed \[[61](#bib.bib61){.ltx_ref}, [62](#bib.bib62){.ltx_ref}, [63](#bib.bib63){.ltx_ref}\], operations are increasingly bottlenecked
by memory (HBM) accesses. Thus exploiting fast SRAM becomes more important.

Execution Model. GPUs have a massive number of threads to execute an operation (called a kernel). Each kernel loads inputs from HBM to registers and SRAM,
computes, then writes outputs to HBM.

Performance characteristics. Depending on the balance of computation and memory accesses, operations can be classified as either compute-bound or memory-bound.
This is commonly measured by the *arithmetic intensity* \[[85](#bib.bib85){.ltx_ref}\], which is the number of arithmetic operations per byte of memory access.

1.  1.
    Compute-bound: the time taken by the operation is determined by how many arithmetic operations there are, while time accessing HBM is much smaller. Typical
    examples are matrix multiply with large inner dimension, and convolution with large number of channels.
2.  2.
    Memory-bound: the time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples
    include most other operations: elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).

Kernel fusion. The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the
input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise
operations \[[53](#bib.bib53){.ltx_ref}, [65](#bib.bib65){.ltx_ref}, [75](#bib.bib75){.ltx_ref}\]. However, in the context of model training, the intermediate
values still need to be written to HBM to save for the backward pass, reducing the effectiveness of naive kernel fusion.

### 2.2 Standard Attention Implementation

Given input sequences ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ where $N$ is the sequence length and $d$ is the head dimension, we want
to compute the attention output $\mathbf{O} \in {\mathbb{R}}^{N \times d}$:

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{\mathbf{S} = {\mathbf{Q}\mathbf{K}}^{\top} \in {\mathbb{R}}^{N \times N}},{\mathbf{P} = {{softmax}\operatorname{}{(\mathbf{S})}} \in {\mathbb{R}}^{N \times N}},{\mathbf{O} = {\mathbf{P}\mathbf{V}} \in {\mathbb{R}}^{N \times d}}},$$   
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

where $softmax$ is applied row-wise.

Standard attention implementations materialize the matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM, which takes $O\operatorname{}{(N^{2})}$ memory. Often
$N \gg d$ (e.g., for GPT2, $N = 1024$ and $d = 64$). We describe the standard attention implementation
in [Algorithm ](#alg0 "Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
As some or most of the operations are memory-bound (e.g., softmax), the large number of memory accesses translates to slow wall-clock time.

This problem is exacerbated by other elementwise operations applied to the attention matrix, such as masking applied to $\mathbf{S}$ or dropout applied to
$\mathbf{P}$. As a result, there have been many attempts to fuse several elementwise operations, such as fusing masking with
softmax \[[77](#bib.bib77){.ltx_ref}\].

In
[Section 3.2](#S3.SS2 "3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
we will show that the standard attention implementation performs HBM accesses quadratic in the sequence length $N$. We also compare the number of FLOPs and
number of HBM accesses of standard attention and of our method (FlashAttention).

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ in HBM.

1:   Load $\mathbf{Q},\mathbf{K}$ by blocks from HBM, compute $\mathbf{S} = {\mathbf{Q}\mathbf{K}}^{\top}$, write $\mathbf{S}$ to HBM.

2:   Read $\mathbf{S}$ from HBM, compute $\mathbf{P} = {{softmax}\operatorname{}{(\mathbf{S})}}$, write $\mathbf{P}$ to HBM.

3:   Load $\mathbf{P}$ and $\mathbf{V}$ by blocks from HBM, compute $\mathbf{O} = {\mathbf{P}\mathbf{V}}$, write $\mathbf{O}$ to HBM.

4:  Return $\mathbf{O}$.

Algorithm 0 Standard Attention Implementation

## 3 FlashAttention: Algorithm, Analysis, and Extensions

We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an
attention algorithm that is both memory efficient and faster in wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM
accesses compared to standard attention. We further show that FlashAttention can serve as a useful primitive by extending it to handle block-sparse attention.

We focus here on the forward pass for ease of exposition;
[Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref} contains details for
the backward.

### 3.1 An Efficient Attention Algorithm With Tiling and Recomputation

Given the inputs ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ in HBM, we aim to compute the attention output
$\mathbf{O} \in {\mathbb{R}}^{N \times d}$ and write it to HBM. Our goal is to reduce the amount of HBM accesses (to sub-quadratic in $N$).

We apply two established techniques (tiling, recomputation) to overcome the technical challenge of computing exact attention in sub-quadratic HBM accesses. We
describe this
in [Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
The main idea is that we split the inputs $\mathbf{Q},\mathbf{K},\mathbf{V}$ into blocks, load them from slow HBM to fast SRAM, then compute the attention
output with respect to those blocks. By scaling the output of each block by the right normalization factor before adding them up, we get the correct result at
the end.

Tiling. We compute attention by blocks. Softmax couples columns of $\mathbf{K}$, so we decompose the large softmax with scaling \[[60](#bib.bib60){.ltx_ref},
[51](#bib.bib51){.ltx_ref}, [66](#bib.bib66){.ltx_ref}\]. For numerical stability, the softmax of vector $x \in {\mathbb{R}}^{B}$ is computed as:

  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{{m\operatorname{}{(x)}}:={\max\limits_{i}\mspace{21mu} x_{i}}},{{{f\operatorname{}{(x)}}:=\begin{bmatrix}                                                                                
     e^{x_{1} - {m\operatorname{}{(x)}}} & \ldots & e^{x_{B} - {m\operatorname{}{(x)}}}                                                                                                           
     \end{bmatrix}},{{{\ell\operatorname{}{(x)}}:={\sum\limits_{i}{f\operatorname{}{(x)}_{i}}}},{{{softmax}\operatorname{}{(x)}}:=\frac{f\operatorname{}{(x)}}{\ell\operatorname{}{(x)}}}}}}.$$   
  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

For vectors ${x^{(1)},x^{(2)}} \in {\mathbb{R}}^{B}$, we can decompose the softmax of the concatenated $x = \begin{bmatrix}
{x^{(1)}\operatorname{}x^{(2)}}
\end{bmatrix} \in {\mathbb{R}}^{2\operatorname{}B}$ as:

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     ${{{m\operatorname{}{(x)}} = {m\operatorname{}{(\begin{bmatrix}                                                                                                                                                                                                                                                                               
     {x^{(1)}\operatorname{}x^{(2)}}                                                                                                                                                                                                                                                                                                               
     \end{bmatrix})}} = {\max{({m\operatorname{}{(x^{(1)})}},{m\operatorname{}{(x^{(2)})}})}}},{{f\operatorname{}{(x)}} = \begin{bmatrix}                                                                                                                                                                                                          
     {e^{{m\operatorname{}{(x^{(1)})}} - {m\operatorname{}{(x)}}}\operatorname{}f\operatorname{}{(x^{(1)})}} & {e^{{m\operatorname{}{(x^{(2)})}} - {m\operatorname{}{(x)}}}\operatorname{}f\operatorname{}{(x^{(2)})}}                                                                                                                             
     \end{bmatrix}}},$                                                                                                                                                                                                                                                                                                                             
     ${{{\ell\operatorname{}{(x)}} = {\ell\operatorname{}{(\begin{bmatrix}                                                                                                                                                                                                                                                                         
     {x^{(1)}\operatorname{}x^{(2)}}                                                                                                                                                                                                                                                                                                               
     \end{bmatrix})}} = {{e^{{m\operatorname{}{(x^{(1)})}} - {m\operatorname{}{(x)}}}\operatorname{}\ell\operatorname{}{(x^{(1)})}} + {e^{{m\operatorname{}{(x^{(2)})}} - {m\operatorname{}{(x)}}}\operatorname{}\ell\operatorname{}{(x^{(2)})}}}},{{{softmax}\operatorname{}{(x)}} = \frac{f\operatorname{}{(x)}}{\ell\operatorname{}{(x)}}}}.$   
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

Therefore if we keep track of some extra statistics (${m\operatorname{}{(x)}},{\ell\operatorname{}{(x)}}$), we can compute softmax one block at a
time.^2^^2^2This style of aggregation is called *algebraic aggregation* \[[33](#bib.bib33){.ltx_ref}\]. We thus split the inputs
$\mathbf{Q},\mathbf{K},\mathbf{V}$ into blocks
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[3](#alg1.l3 "3 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}),
compute the softmax values along with extra statistics
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[10](#alg1.l10 "10 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}),
and combine the results
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[12](#alg1.l12 "12 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).

Recomputation. One of our goals is to not store $O\operatorname{}{(N^{2})}$ intermediate values for the backward pass. The backward pass typically requires the
matrices ${\mathbf{S},\mathbf{P}} \in {\mathbb{R}}^{N \times N}$ to compute the gradients with respect to $\mathbf{Q},\mathbf{K},\mathbf{V}$. However, by
storing the output $\mathbf{O}$ and the softmax normalization statistics $(m,\ell)$, we can recompute the attention matrix $\mathbf{S}$ and $\mathbf{P}$ easily
in the backward pass from blocks of $\mathbf{Q},\mathbf{K},\mathbf{V}$ in SRAM. This can be seen as a form of selective gradient
checkpointing \[[34](#bib.bib34){.ltx_ref}, [10](#bib.bib10){.ltx_ref}\]. While gradient checkpointing has been suggested to reduce the maximum amount of memory
required \[[66](#bib.bib66){.ltx_ref}\], all implementations (that we know off) have to trade speed for memory. In contrast, even with more FLOPs, our
recomputation speeds up the backward pass due to reduced HBM accesses
([Fig. 2](#S3.F2 "Figure 2 ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
The full backward pass description is
in [Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

Implementation details: Kernel fusion. Tiling enables us to implement our algorithm in one CUDA kernel, loading input from HBM, performing all the computation
steps (matrix multiply, softmax, optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout
in [Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}). This avoids
repeatedly reading and writing of inputs and outputs from and to HBM.

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ in HBM, on-chip SRAM of size $M$.

1:  Set block sizes
${B_{c} = \left\lceil \frac{M}{4\operatorname{}d} \right\rceil},{B_{r} = {\min\left( \left\lceil \frac{M}{4\operatorname{}d} \right\rceil,d \right)}}$.

2:   Initialize
${\mathbf{O} = {(0)}_{N \times d} \in {\mathbb{R}}^{N \times d}},{\ell = {(0)}_{N} \in {\mathbb{R}}^{N}},{m = {({- \infty})}_{N} \in {\mathbb{R}}^{N}}$ in HBM.

3:   Divide $\mathbf{Q}$ into $T_{r} = \left\lceil \frac{N}{B_{r}} \right\rceil$ blocks $\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}$ of size $B_{r} \times d$
each, and divide $\mathbf{K},\mathbf{V}$ in to $T_{c} = \left\lceil \frac{N}{B_{c}} \right\rceil$ blocks $\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}$ and
$\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}$, of size $B_{c} \times d$ each.

4:  Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}$ of size $B_{r} \times d$ each, divide $\ell$ into $T_{r}$ blocks
$\ell_{i},\ldots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\ldots,m_{T_{r}}$ of size $B_{r}$ each.

5:  for $1 \leq j \leq T_{c}$ do

6:      Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.

7:     for $1 \leq i \leq T_{r}$ do

8:         Load $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.

9:         On chip, compute $\mathbf{S}_{i\operatorname{}j} = {\mathbf{Q}_{i}\operatorname{}\mathbf{K}_{j}^{T}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

10:         On chip, compute ${\overset{\sim}{m}}_{i\operatorname{}j} = {{rowmax}\operatorname{}{(\mathbf{S}_{i\operatorname{}j})}} \in {\mathbb{R}}^{B_{r}}$,
${\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j} = {\exp{({\mathbf{S}_{i\operatorname{}j} - {\overset{\sim}{m}}_{i\operatorname{}j}})}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$
(pointwise),
${\overset{\sim}{\ell}}_{i\operatorname{}j} = {{rowsum}\operatorname{}{({\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j})}} \in {\mathbb{R}}^{B_{r}}$.

11:        On chip, compute $m_{i}^{new} = {\max{(m_{i},{\overset{\sim}{m}}_{i\operatorname{}j})}} \in {\mathbb{R}}^{B_{r}}$,
$\ell_{i}^{new} = {{e^{m_{i} - m_{i}^{new}}\operatorname{}\ell_{i}} + {e^{{\overset{\sim}{m}}_{i\operatorname{}j} - m_{i}^{new}}\operatorname{}{\overset{\sim}{\ell}}_{i\operatorname{}j}}} \in {\mathbb{R}}^{B_{r}}$.

12:         Write
$\mathbf{O}_{i}\leftarrow{{diag}\operatorname{}{(\ell_{i}^{new})}^{- 1}\operatorname{}{({{{diag}\operatorname{}{(\ell_{i})}\operatorname{}e^{m_{i} - m_{i}^{new}}\operatorname{}\mathbf{O}_{i}} + {e^{{\overset{\sim}{m}}_{i\operatorname{}j} - m_{i}^{new}}\operatorname{}{\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j}\operatorname{}\mathbf{V}_{j}}})}}$
to HBM.

13:        Write $\ell_{i}\leftarrow\ell_{i}^{new}$, $m_{i}\leftarrow m_{i}^{new}$ to HBM.

14:     end for

15:  end for

16:  Return $\mathbf{O}$.

Algorithm 1 FlashAttention

We show FlashAttention's correctness, runtime, and memory requirement (proof
in [Appendix C](#A3 "Appendix C Proofs ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).

###### Theorem 1.

[Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref
.ltx_font_italic} returns $\mathbf{O} = {{softmax}\operatorname{}{({\mathbf{Q}\mathbf{K}}^{\top})}\operatorname{}\mathbf{V}}$ with
$O\operatorname{}{({N^{2}\operatorname{}d})}$ FLOPs and requires $O\operatorname{}{(N)}$ additional memory beyond inputs and output.

### 3.2 Analysis: IO Complexity of FlashAttention

We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound,
proving that no exact attention algorithm can asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in
[Appendix C](#A3 "Appendix C Proofs ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

###### Theorem 2.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d \leq M \leq {N\operatorname{}d}$. Standard attention
([Algorithm ](#alg0 "Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
requires $\Theta\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ HBM accesses, while FlashAttention
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
requires $\Theta\operatorname{}{({N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}})}$ HBM accesses.

For typical values of $d$ (64-128) and $M$ (around 100KB), $d^{2}$ is many times smaller than $M$, and thus FlashAttention requires many times fewer HBM
accesses than standard implementation. This leads to both faster execution and lower memory footprint, which we validate
in [Section 4.3](#S4.SS3 "4.3 Benchmarking Attention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

The main idea of the proof is that given the SRAM size of $M$, we can load blocks of $\mathbf{K},\mathbf{V}$ of size $\Theta\operatorname{}{(M)}$ each
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[6](#alg1.l6 "6 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
For each block of $\mathbf{K}$ and $\mathbf{V}$, we iterate over all blocks of $\mathbf{Q}$
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[8](#alg1.l8 "8 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
to compute the intermediate values, resulting in $\Theta\operatorname{}{({N\operatorname{}d\operatorname{}M^{- 1}})}$ passes over $\mathbf{Q}$. Each pass loads
$\Theta\operatorname{}{({N\operatorname{}d})}$ elements, which amounts to $\Theta\operatorname{}{({N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}})}$ HBM
accesses. We similarly prove that the backward pass of standard attention requires $\Theta\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ HBM accesses while
the backward pass of FlashAttention requires $\Theta\operatorname{}{({N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}})}$ HBM accesses
([Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).

We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all values of $M$ (the SRAM size) when computing exact attention.

###### Proposition 3.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d \leq M \leq {N\operatorname{}d}$. There does not exist an algorithm
to compute exact attention with $o\operatorname{}{({N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}})}$ HBM accesses for all $M$ in the range
$\lbrack d,{N\operatorname{}d}\rbrack$.

The proof relies on the fact that for $M = {\Theta\operatorname{}{({N\operatorname{}d})}}$ any algorithm must perform
${\Omega\operatorname{}{({N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}})}} = {\Omega\operatorname{}{({N\operatorname{}d})}}$ HBM accesses. This type of lower
bound over a subrange of $M$ is common in the streaming algorithms literature \[[88](#bib.bib88){.ltx_ref}\]. We leave proving parameterized
complexity \[[27](#bib.bib27){.ltx_ref}\] lower bounds in terms of $M$ as exciting future work.

We validate that the number of HBM accesses is the main determining factor of attention run-time.
In [Fig. 2](#S3.F2 "Figure 2 ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
(left), we see that even though FlashAttention has higher FLOP count compared to standard attention (due to recomputation in the backward pass), it has much
fewer HBM accesses, resulting in much faster runtime.
In [Fig. 2](#S3.F2 "Figure 2 ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
(middle), we vary the block size $B_{c}$ of FlashAttention, which results in different amounts of HBM accesses, and measure the runtime of the forward pass. As
block size increases, the number of HBM accesses decreases (as we make fewer passes over the input), and runtime decreases. For large enough block size (beyond
256), the runtime is then bottlenecked by other factors (e.g., arithmetic operations). Moreover, larger block size will not fit into the small SRAM size.

  Attention      Standard   FlashAttention
  -------------- ---------- ----------------
  GFLOPs         66.6       75.2
  HBM R/W (GB)   40.3       4.4
  Runtime (ms)   41.7       7.3

![Refer to caption](/html/2205.14135/assets/x2.png){#S3.F2.2.p1.g1 .ltx_graphics .ltx_img_landscape width="415" height="140"}

Figure 2: Left: Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium (seq. length 1024, head dim. 64, 16 heads, batch size 64)
on A100 GPU. HBM access is the primary factor affecting runtime. Middle: Forward runtime of FlashAttention (seq. length 1024, head dim. 64, 16 heads, batch size
64) on A100 GPU. Fewer HBM accesses result in faster runtime, up to a point. Right: The runtime (for seq. length 4K) of block-sparse FlashAttention is faster
than FlashAttention by a factor proportional to the sparsity.

### 3.3 Extension: Block-Sparse FlashAttention

We extend FlashAttention to approximate attention: we propose block-sparse FlashAttention, whose IO complexity is smaller than FlashAttention by a factor
proportional to the sparsity.

Given inputs ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ and a mask matrix $\overset{\sim}{\mathbf{M}} \in {\{ 0,1\}}^{N \times N}$, we
want to compute:

  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{\mathbf{S} = {\mathbf{Q}\mathbf{K}}^{\top} \in {\mathbb{R}}^{N \times N}},{\mathbf{P} = {{softmax}\operatorname{}{({{\mathbf{S} \odot}\operatorname{}1_{\overset{\sim}{\mathbf{M}}}})}} \in {\mathbb{R}}^{N \times N}},{\mathbf{O} = {\mathbf{P}\mathbf{V}} \in {\mathbb{R}}^{N \times d}}},$$   
  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

where ${({{\mathbf{S} \odot}\operatorname{}1_{\overset{\sim}{\mathbf{M}}}})}_{k\operatorname{}l} = \mathbf{S}_{k\operatorname{}l}$ if
${\overset{\sim}{\mathbf{M}}}_{k\operatorname{}l} = 1$ and $- \infty$ if $\mathbf{M}_{k\operatorname{}l} = 0$. We require $\overset{\sim}{\mathbf{M}}$ to have
block form: for some block sizes $B_{r},B_{c}$, for all $k,l$, ${\overset{\sim}{\mathbf{M}}}_{k,l} = \mathbf{M}_{i\operatorname{}j}$ with
${i = {\lfloor{k/B_{r}}\rfloor}},{j = {\lfloor{l/B_{c}}\rfloor}}$ for some $\mathbf{M} \in {\{ 0,1\}}^{{{N/B_{r}} \times N}/B_{c}}$.

Given a predefined block sparsity mask $\mathbf{M} \in {\{ 0,1\}}^{{{N/B_{r}} \times N}/B_{c}}$ we can easily
adapt [Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
to only compute the nonzero blocks of the attention matrix. The algorithm is identical
to [Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
except we skip zero blocks. We reproduce the algorithm description
in [Algorithm 5](#alg5 "Algorithm 5 ‣ D.1 Block-sparse FlashAttention ‣ Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
in [Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

We also analyze the IO complexity of block-sparse FlashAttention.

###### Proposition 4.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d \leq M \leq {N\operatorname{}d}$. Block-sparse FlashAttention
([Algorithm 5](#alg5 "Algorithm 5 ‣ D.1 Block-sparse FlashAttention ‣ Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
requires $\Theta\operatorname{}{({{N\operatorname{}d} + {N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}\operatorname{}s}})}$ HBM accesses where $s$ is the
fraction of nonzero blocks in the block-sparsity mask.

We see that applying block-sparsity yields a direct improvement by the sparsity to the larger term in the IO complexity. For large sequence lengths $N$, $s$ is
often set to $N^{- {1/2}}$ \[[11](#bib.bib11){.ltx_ref}\] or $N^{- 1}\operatorname{}{\log N}$ \[[92](#bib.bib92){.ltx_ref}, [3](#bib.bib3){.ltx_ref},
[17](#bib.bib17){.ltx_ref}\], resulting in $\Theta\operatorname{}{({N\operatorname{}\sqrt{N}})}$ or $\Theta\operatorname{}{({N\operatorname{}{\log N}})}$ IO
complexity. For downstream experiments, we use the fixed butterfly sparsity pattern \[[17](#bib.bib17){.ltx_ref}\], which has been shown to be able to
approximate arbitrary sparsity \[[16](#bib.bib16){.ltx_ref}\].

In [Fig. 2](#S3.F2 "Figure 2 ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
(right), we validate that as the sparsity increases, the runtime of block-sparse FlashAttention improves proportionally. On the LRA benchmark, block-sparse
FlashAttention achieves 2.8$\times$ speedup, while performing on par with standard attention
([Section 4](#S4 "4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).

## 4 Experiments

We evaluate the impact of using FlashAttention to train Transformer models. We validate two claims about training time and model accuracy, and report attention
runtime and memory benchmarks.

-   •
    Training Speed. FlashAttention outperforms the MLPerf 1.1 \[[58](#bib.bib58){.ltx_ref}\] speed record for BERT by 15%, and speeds up GPT-2 up to 3$\times$
    over HuggingFace \[[87](#bib.bib87){.ltx_ref}\] and $1.8 \times$ over Megatron \[[77](#bib.bib77){.ltx_ref}\] over standard Transformers. FlashAttention
    speeds up the long-range arena (LRA) benchmark 2.4$\times$.
-   •
    Quality. FlashAttention scales Transformers to longer sequences, yielding higher quality. FlashAttention trains GPT-2 with context length 4K faster than
    Megatron trains GPT-2 with context length 1K, while achieving 0.7 better perplexity. Modeling longer sequences yields 6.4 points of lift on two
    long-document classification tasks. Finally, FlashAttention yields the first Transformer that can achieve better-than-random performance on the challenging
    Path-X task (sequence length 16K), and block-sparse FlashAttention yields the first sequence model that we know of that can achieve better-than-random
    performance on Path-256 (sequence length 64K).
-   •
    Benchmarking Attention. We measure the runtime and memory performance of FlashAttention and block-sparse FlashAttention based on sequence length. We confirm
    that the memory footprint of FlashAttention scales linearly with seq. length and is up to 3$\times$ faster than standard attention for common seq. lengths
    (up to 2K). We confirm that runtime of block-sparse FlashAttention scales linearly in seq. length and is faster than all existing approximate attention
    baselines.

Additional experiment details are
in [Appendix E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

### 4.1 Faster Models with FlashAttention

##### BERT.

FlashAttention yields the fastest single-node BERT training speed that we know of. We train a BERT-large \[[22](#bib.bib22){.ltx_ref}\] model with
FlashAttention on Wikipedia.
[Table 1](#S4.T1 "Table 1 ‣ BERT. ‣ 4.1 Faster Models with FlashAttention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
compares our training time to the implementation from Nvidia that set the training speed record for MLPerf 1.1 \[[58](#bib.bib58){.ltx_ref}\]. Our
implementation is 15% faster.

  BERT Implementation                                Training time (minutes)
  -------------------------------------------------- -------------------------
  Nvidia MLPerf 1.1 \[[58](#bib.bib58){.ltx_ref}\]   20.0 $\pm$ 1.5
  FlashAttention (ours)                              17.4 $\pm$ 1.4

Table 1: Training time of BERT-large, starting from the same initialization provided by the MLPerf benchmark, to reach the target accuracy of 72.0% on masked
language modeling. Averaged over 10 runs on 8$\times$A100 GPUs.

##### GPT-2.

FlashAttention yields faster training times for GPT-2 \[[67](#bib.bib67){.ltx_ref}\] on the large OpenWebtext dataset \[[32](#bib.bib32){.ltx_ref}\] than the
widely used HuggingFace \[[87](#bib.bib87){.ltx_ref}\] and Megatron-LM \[[77](#bib.bib77){.ltx_ref}\] implementations.
Table [2](#S4.T2 "Table 2 ‣ GPT-2. ‣ 4.1 Faster Models with FlashAttention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows up to 3$\times$ end-to-end speedup compared to Huggingface and 1.7$\times$ speedup compared to Megatron-LM. FlashAttention achieves the same perplexity as
the other two implementations, as we do not change the model definition.
[Appendix E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref} includes plots
of the validation perplexity throughout training, confirming that FlashAttention is as numerically stable as the baselines and produces the same training /
validation curves.

  Model implementations                                       OpenWebText (ppl)   Training time (speedup)
  ----------------------------------------------------------- ------------------- -------------------------
  GPT-2 small - Huggingface \[[87](#bib.bib87){.ltx_ref}\]    18.2                9.5 days (1.0$\times$)
  GPT-2 small - Megatron-LM \[[77](#bib.bib77){.ltx_ref}\]    18.2                4.7 days (2.0$\times$)
  GPT-2 small - FlashAttention                                18.2                2.7 days (3.5$\times$)
  GPT-2 medium - Huggingface \[[87](#bib.bib87){.ltx_ref}\]   14.2                21.0 days (1.0$\times$)
  GPT-2 medium - Megatron-LM \[[77](#bib.bib77){.ltx_ref}\]   14.3                11.5 days (1.8$\times$)
  GPT-2 medium - FlashAttention                               14.3                6.9 days (3.0$\times$)

Table 2: GPT-2 small and medium using FlashAttention achieve up to 3$\times$ speed up compared to Huggingface implementation and up to 1.7$\times$ compared to
Megatron-LM. Training time reported on 8$\times$A100s GPUs.

##### Long-range Arena.

We compare vanilla Transformer (with either standard implementation or FlashAttention) on the long-range arena (LRA \[[80](#bib.bib80){.ltx_ref}\]) benchmark.
We measure accuracy, throughput, and training time of all models. Each task has a different sequence length varying between 1024 and 4096. We follow the
implementation and experimental setting in Tay et al. \[[80](#bib.bib80){.ltx_ref}\]and Xiong et al. \[[90](#bib.bib90){.ltx_ref}\].^3^^3^3LRA accuracy results
are known to be highly dependent on the tuning procedure \[[90](#bib.bib90){.ltx_ref}\]. Our reproduced baselines perform better than as reported in the
original comparison \[[80](#bib.bib80){.ltx_ref}\].
[Table 3](#S4.T3 "Table 3 ‣ Long-range Arena. ‣ 4.1 Faster Models with FlashAttention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows that FlashAttention achieves up 2.4$\times$ speed-up compared to standard attention. Block-sparse FlashAttention is faster than all of the approximate
attention methods that we have tested.

  Models                                            ListOps   Text   Retrieval   Image   Pathfinder   Avg    Speedup
  ------------------------------------------------- --------- ------ ----------- ------- ------------ ------ -------------
  Transformer                                       36.0      63.6   81.6        42.3    72.7         59.3   -
  FlashAttention                                    37.6      63.9   81.4        43.5    72.7         59.8   2.4$\times$
  Block-sparse FlashAttention                       37.0      63.0   81.3        43.6    73.3         59.6   2.8$\times$
  Linformer \[[84](#bib.bib84){.ltx_ref}\]          35.6      55.9   77.7        37.8    67.6         54.9   2.5$\times$
  Linear Attention \[[50](#bib.bib50){.ltx_ref}\]   38.8      63.2   80.7        42.6    72.5         59.6   2.3$\times$
  Performer \[[12](#bib.bib12){.ltx_ref}\]          36.8      63.6   82.2        42.1    69.9         58.9   1.8$\times$
  Local Attention \[[80](#bib.bib80){.ltx_ref}\]    36.1      60.2   76.7        40.6    66.6         56.0   1.7$\times$
  Reformer \[[51](#bib.bib51){.ltx_ref}\]           36.5      63.8   78.5        39.6    69.4         57.6   1.3$\times$
  Smyrf \[[19](#bib.bib19){.ltx_ref}\]              36.1      64.1   79.0        39.6    70.5         57.9   1.7$\times$

Table 3: The performance of standard attention, FlashAttention, block-sparse FlashAttention, and approximate attention baselines on the Long-Range-Arena
benchmarks.

### 4.2 Better Models with Longer Sequences

##### Language Modeling with Long Context.

The runtime and memory-efficiency of FlashAttention allow us to increase the context length of GPT-2 by 4$\times$ while still running faster than the optimized
implementation from Megatron-LM.
[Table 4](#S4.T4 "Table 4 ‣ Language Modeling with Long Context. ‣ 4.2 Better Models with Longer Sequences ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows that that GPT-2 with FlashAttention and context length 4K is still 30% faster than GPT-2 from Megatron with context length 1K, while achieving 0.7 better
perplexity.

  Model implementations          Context length   OpenWebText (ppl)   Training time (speedup)
  ------------------------------ ---------------- ------------------- -------------------------
  GPT-2 small - Megatron-LM      1k               18.2                4.7 days (1.0$\times$)
  GPT-2 small - FlashAttention   1k               18.2                2.7 days (1.7$\times$)
  GPT-2 small - FlashAttention   2k               17.6                3.0 days (1.6$\times$)
  GPT-2 small - FlashAttention   4k               17.5                3.6 days (1.3$\times$)

Table 4: GPT-2 small with FlashAttention, with 4$\times$ larger context length compared to Megatron-LM, is still 30% faster while achieving 0.7 better
perplexity. Training time on 8$\times$A100 GPUs is reported.

##### Long Document Classification.

Training Transformers with longer sequences with FlashAttention improves performance on the MIMIC-III \[[47](#bib.bib47){.ltx_ref}\] and
ECtHR \[[6](#bib.bib6){.ltx_ref}, [7](#bib.bib7){.ltx_ref}\] datasets. MIMIC-III contains intensive care unit patient discharge summaries, each annotated with
multiple labels. ECtHR contains legal cases from the European Court of Human Rights, each of which is mapped to articles of the Convention of Human Rights that
were allegedly violaged. Both of these datasets contain very long text documents; the average number of tokens in MIMIC is 2,395 tokens, and the longest
document contains 14,562 tokens, while the average and longest numbers in ECtHR are 2,197 and 49,392, respectively. We evaluate lift from increasing the
sequence length of a pretrained RoBERTa model \[[56](#bib.bib56){.ltx_ref}\] (we repeat the positional embeddings, as in Beltagy et al.
\[[3](#bib.bib3){.ltx_ref}\]).

Table [6](#S4.T6 "Table 6 ‣ Long Document Classification. ‣ 4.2 Better Models with Longer Sequences ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows that sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and that length 8K outperforms length 512 by 8.5 points on ECtHR. The
discrepancies may be due to subtle distribution shifts: MIMIC-III contains specialized medical text and thus may be more susceptible to a distribution shift in
the document length, whereas ECtHR contains general language.

                                             512    1024   2048   4096   8192   16384
  ------------------------------------------ ------ ------ ------ ------ ------ -------
  MIMIC-III \[[47](#bib.bib47){.ltx_ref}\]   52.8   50.7   51.7   54.6   56.4   57.1
  ECtHR \[[6](#bib.bib6){.ltx_ref}\]         72.2   74.3   77.1   78.6   80.7   79.2

  ------------------------------------------------- -------- ----------
  Model                                             Path-X   Path-256
  Transformer                                       ✗        ✗
  Linformer \[[84](#bib.bib84){.ltx_ref}\]          ✗        ✗
  Linear Attention \[[50](#bib.bib50){.ltx_ref}\]   ✗        ✗
  Performer \[[12](#bib.bib12){.ltx_ref}\]          ✗        ✗
  Local Attention \[[80](#bib.bib80){.ltx_ref}\]    ✗        ✗
  Reformer \[[51](#bib.bib51){.ltx_ref}\]           ✗        ✗
  SMYRF \[[19](#bib.bib19){.ltx_ref}\]              ✗        ✗
  FlashAttention                                    61.4     ✗
  Block-sparse FlashAttention                       56.0     63.1
  ------------------------------------------------- -------- ----------

Table 5: Long Document performance (micro $F_{1}$) at different sequence lengths using FlashAttention.

Table 6: We report the first Transformer model that can achieve non-random performance on Path-X and Path-256.

Table 6: We report the first Transformer model that can achieve non-random performance on Path-X and Path-256.

##### Path-X and Path-256.

The Path-X and Path-256 benchmarks are challenging tasks from the long-range arena benchmark designed to test long context. The task is to classify whether two
points in a black and white 128$\times$128 (or 256$\times$256) image have a path connecting them, and the images are fed to the transformer one pixel at a time.
In prior work, all transformer models have either run out of memory, or only achieved random performance \[[80](#bib.bib80){.ltx_ref}\]. There has been a search
for alternative architectures that can model such long context \[[37](#bib.bib37){.ltx_ref}\]. We present here the first result of Transformer models being able
to solve Path-X and Path-256
([Table 6](#S4.T6 "Table 6 ‣ Long Document Classification. ‣ 4.2 Better Models with Longer Sequences ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
We pretrain a transformer on Path-64, and then transfer to Path-X by spatially interpolating the positional embeddings. FlashAttention achieves 61.4 accuracy on
Path-X. Additionally, block-sparse FlashAttention enables the Transformers to scale to sequence length 64K, achieving 63.1 accuracy^4^^4^4Path-256 requires
longer sequences but has relatively shorter paths than Path-X, so it is easier to obtain a higher accuracy. on Path-256.

### 4.3 Benchmarking Attention

![Refer to caption](/html/2205.14135/assets/x3.png){#S4.F3.g1 .ltx_graphics .ltx_centering .ltx_img_landscape width="761" height="232"}

Figure 3: Left: runtime of forward pass + backward pass. Right: attention memory usage.

We vary sequence length and measure runtime and memory usage of FlashAttention and block-sparse FlashAttention against various attention baselines on one A100
GPU with 40 GB HBM, with dropout and a padding mask. We compare against reference implementations for exact attention, approximate attention, and sparse
attention. We report a subset of baselines in the main body;
Appendix [E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref} contains more
baselines and full details.

##### Runtime.

Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Benchmarking Attention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
(left) reports the runtime in milliseconds of the forward + backward pass of FlashAttention and block-sparse FlashAttention compared to the baselines in exact,
approximate, and sparse attention (exact numbers in
Appendix [E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}). Runtime grows
quadratically with sequence length, but FlashAttention runs significantly faster than exact attention baselines, up to 3$\times$ faster than the PyTorch
implementation. The runtimes of many approximate/sparse attention mechanisms grow linearly with sequence length, but FlashAttention still runs faster than
approximate and sparse attention for short sequences due to fewer memory accesses. The approximate attention runtimes begin to cross over with FlashAttention at
sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention
that we know of, across all sequence lengths.

##### Memory Footprint.

Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Benchmarking Attention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
(right) shows the memory footprint of FlashAttention and block-sparse FlashAttention compared to various exact, approximate, and sparse attention baselines.
FlashAttention and block-sparse FlashAttention have the same memory footprint, which grows linearly with sequence length. FlashAttention is up to 20$\times$
more memory efficient than exact attention baselines, and is more memory-efficient than the approximate attention baselines. All other algorithms except for
Linformer run out of memory on an A100 GPU before 64K, and FlashAttention is still 2$\times$ more efficient than Linformer.

## 5 Limitations and Future Directions

We discuss limitations of our approach and future directions. Related work is given
in [Appendix A](#A1 "Appendix A Related Work ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

Compiling to CUDA. Our current approach to building IO-aware implementations of attention requires writing a new CUDA kernel for each new attention
implementation. This requires writing the attention algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort.
Implementations may also not be transferrable across GPU architectures. These limitations suggest the need for a method that supports writing attention
algorithms in a high-level language (e.g., PyTorch), and compiling to IO-aware implementations in CUDA---similar to efforts such as Halide in image
processing \[[70](#bib.bib70){.ltx_ref}\].

IO-Aware Deep Learning. We believe that the IO-aware approach can extend beyond attention. Attention is the most memory-intensive computation in Transformers,
but every layer in a deep network touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss these potential
extensions in [Appendix D](#A4 "Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

Multi-GPU IO-Aware Methods. Our IO-aware implementation of attention is optimal within constants for computing attention on a single GPU. However, the attention
computation may be parallelizable across multiple GPUs \[[72](#bib.bib72){.ltx_ref}\]. Using multiple GPUs adds an additional layer to IO analysis---accounting
for data transfer between GPUs. We hope our work inspires future work in this direction.

#### Acknowledgments

Our implementation uses Apex's FMHA code
([https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha){.ltx_ref .ltx_url
.ltx_font_typewriter target="_blank"}) as a starting point. We thank Young-Jun Ko for the in-depth explanation of his FMHA implementation and for his thoughtful
answers to our questions about CUDA. We thank Sabri Eyuboglu, Megan Leszczynski, Laurel Orr, Yuhuai Wu, Beidi Chen, and Xun Huang for their constructive
feedback and suggestions on early drafts of the paper. We thank Markus Rabe and Charles Staats for helpful discussion of their attention algorithm.

We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and
1937301 (RTML); ARL under No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266 (Unifying Weak Supervision); ONR N00014-20-1-2480:
Understanding and Applying Non-Euclidean Geometry in Machine Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba,
TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP & HAI-Azure Cloud Credits for Research
program, the Stanford Data Science Initiative (SDSI), Department of Defense (DoD) through the National Defense Science and Engineering Graduate Fellowship
(NDSEG) Program, and members of the Stanford DAWN project: Facebook, Google, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints
for Governmental purposes notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material
are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government.
Atri Rudra's research is supported by NSF grant CCF-1763481.

## References

-   Aggarwal and Vitter \[1988\] Alok Aggarwal and S Vitter, Jeffrey. The input/output complexity of sorting and related problems. *Communications of the ACM*,
    31(9):1116--1127, 1988.
-   Bello \[2021\] Irwan Bello. LambdaNetworks: Modeling long-range interactions without attention. *arXiv preprint arXiv:2102.08602*, 2021.
-   Beltagy et al. \[2020\] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020.
-   Blackford et al. \[2002\] L Susan Blackford, Antoine Petitet, Roldan Pozo, Karin Remington, R Clint Whaley, James Demmel, Jack Dongarra, Iain Duff, Sven
    Hammarling, Greg Henry, et al. An updated set of basic linear algebra subprograms (blas). *ACM Transactions on Mathematical Software*, 28(2):135--151, 2002.
-   Brown et al. \[2020\] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish
    Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877--1901, 2020.
-   Chalkidis et al. \[2019\] Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. Neural legal judgment prediction in English. In *Proceedings of the
    57th Annual Meeting of the Association for Computational Linguistics*, pages 4317--4323, Florence, Italy, 2019. Association for Computational Linguistics.
    doi: [10.18653/v1/P19-1424](10.18653/v1/P19-1424){.ltx_ref .ltx_Url}. URL
    [https://www.aclweb.org/anthology/P19-1424](https://www.aclweb.org/anthology/P19-1424){.ltx_ref .ltx_url .ltx_font_typewriter target="_blank"}.
-   Chalkidis et al. \[2021\] Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodromos Malakasiotis.
    Paragraph-level rationale extraction through regularization: A case study on european court of human rights cases. In *Proceedings of the Annual Conference
    of the North American Chapter of the Association for Computational Linguistics*, Mexico City, Mexico, 2021. Association for Computational Linguistics.
-   Charlier et al. \[2021\] Benjamin Charlier, Jean Feydy, Joan Alexis Glaunès, François-David Collin, and Ghislain Durif. Kernel operations on the gpu, with
    autodiff, without memory overflows. *Journal of Machine Learning Research*, 22(74):1--6, 2021. URL
    [http://jmlr.org/papers/v22/20-275.html](http://jmlr.org/papers/v22/20-275.html){.ltx_ref .ltx_url .ltx_font_typewriter target="_blank"}.
-   Chen et al. \[2021\] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying sparse and low-rank attention. In
    *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.
-   Chen et al. \[2016\] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. *arXiv preprint
    arXiv:1604.06174*, 2016.
-   Child et al. \[2019\] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv preprint
    arXiv:1904.10509*, 2019.
-   Choromanski et al. \[2020\] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins,
    Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In *International Conference on Learning Representations
    (ICLR)*, 2020.
-   Dai et al. \[2022\] Xiang Dai, Ilias Chalkidis, Sune Darkner, and Desmond Elliott. Revisiting transformer-based models for long document classification.
    *arXiv preprint arXiv:2204.06683*, 2022.
-   Dai et al. \[2019\] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models
    beyond a fixed-length context. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 2978--2988, 2019.
-   Dao et al. \[2019\] Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, and Christopher Ré. Learning fast algorithms for linear transforms using butterfly
    factorizations. In *International Conference on Machine Learning (ICML)*, 2019.
-   Dao et al. \[2020\] Tri Dao, Nimit Sohoni, Albert Gu, Matthew Eichhorn, Amit Blonder, Megan Leszczynski, Atri Rudra, and Christopher Ré. Kaleidoscope: An
    efficient, learnable representation for all structured linear maps. In *International Conference on Learning Representations (ICLR)*, 2020.
-   Dao et al. \[2022a\] Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra, and Christopher Ré. Pixelated butterfly: Simple and efficient
    sparse training for neural network models. In *International Conference on Learning Representations (ICLR)*, 2022a.
-   Dao et al. \[2022b\] Tri Dao, Beidi Chen, Nimit Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher
    Ré. Monarch: Expressive structured matrices for efficient and accurate training. In *International Conference on Machine Learning (ICML)*, 2022b.
-   Daras et al. \[2020\] Giannis Daras, Nikita Kitaev, Augustus Odena, and Alexandros G Dimakis. Smyrf-efficient attention using asymmetric clustering.
    *Advances in Neural Information Processing Systems*, 33:6476--6489, 2020.
-   De Sa et al. \[2018\] Christopher De Sa, Albert Gu, Rohan Puttagunta, Christopher Ré, and Atri Rudra. A two-pronged progress in structured dense matrix
    vector multiplication. In *Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms*, pages 1060--1079. SIAM, 2018.
-   Denning \[1968\] Peter J Denning. The working set model for program behavior. *Communications of the ACM*, 11(5):323--333, 1968.
-   Devlin et al. \[2019\] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language
    understanding.  2019.
-   Dong et al. \[2017\] Xin Dong, Shangyu Chen, and Sinno Jialin Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. *arXiv
    preprint arXiv:1705.07565*, 2017.
-   Dosovitskiy et al. \[2020\] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani,
    Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *International
    Conference on Learning Representations*, 2020.
-   Eidelman and Gohberg \[1999\] Y Eidelman and I Gohberg. On a new class of structured matrices. *Integral Equations and Operator Theory*,
    34(3):293--324, 1999.
-   Feydy et al. \[2020\] Jean Feydy, Joan Glaunès, Benjamin Charlier, and Michael Bronstein. Fast geometric learning with symbolic matrices. *Advances in
    Neural Information Processing Systems*, 33, 2020.
-   Flum and Grohe \[2006\] Jörg Flum and Martin Grohe. *Parameterized Complexity Theory*. Springer, 2006.
-   Frankle and Carbin \[2018\] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International
    Conference on Learning Representations*, 2018.
-   Frankle et al. \[2019\] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin. Stabilizing the lottery ticket hypothesis. *arXiv
    preprint arXiv:1903.01611*, 2019.
-   Frankle et al. \[2020\] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket
    hypothesis. In *International Conference on Machine Learning*, pages 3259--3269. PMLR, 2020.
-   Goel et al. \[2022\] Karan Goel, Albert Gu, Chris Donahue, and Christopher Ré. It's raw! audio generation with state-space models. In *International
    Conference on Machine Learning (ICML)*, 2022.
-   Gokaslan et al. \[2019\] Aaron Gokaslan, Vanya Cohen, Pavlick Ellie, and Stefanie Tellex. Openwebtext corpus, 2019.
-   Gray et al. \[1997\] Jim Gray, Surajit Chaudhuri, Adam Bosworth, Andrew Layman, Don Reichart, Murali Venkatrao, Frank Pellow, and Hamid Pirahesh. Data cube:
    A relational aggregation operator generalizing group-by, cross-tab, and sub-totals. *Data mining and knowledge discovery*, 1(1):29--53, 1997.
-   Griewank and Walther \[2008\] Andreas Griewank and Andrea Walther. *Evaluating derivatives: principles and techniques of algorithmic differentiation*.
    SIAM, 2008.
-   Gu et al. \[2020\] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. Hippo: Recurrent memory with optimal polynomial projections. In
    *Advances in neural information processing systems (NeurIPS)*, 2020.
-   Gu et al. \[2021\] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. Combining recurrent, convolutional, and
    continuous-time models with linear state space layers. *Advances in Neural Information Processing Systems*, 34, 2021.
-   Gu et al. \[2022\] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In *The International
    Conference on Learning Representations (ICLR)*, 2022.
-   Han et al. \[2015\] Song Han, Jeff Pool, John Tran, and William J Dally. Learning both weights and connections for efficient neural networks. *arXiv
    preprint arXiv:1506.02626*, 2015.
-   Han et al. \[2016\] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and
    huffman coding. In *International Conference on Learning Representations*, 2016.
-   Hennessy and Patterson \[2003\] John Hennessy and David Patterson. Memory hierarchy design. *Computer Architecture: A Quantitative Approach*, pages
    390--525, 2003.
-   Hooker \[2020\] Sara Hooker. The hardware lottery. *arXiv preprint arXiv:2009.06489*, 2020.
-   Hua et al. \[2022\] Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V Le. Transformer quality in linear time. *arXiv preprint arXiv:2202.10447*, 2022.
-   Ivanov et al. \[2021\] Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, and Torsten Hoefler. Data movement is all you need: A case study on optimizing
    transformers. *Proceedings of Machine Learning and Systems*, 3:711--732, 2021.
-   Jia and Van Sandt \[2021\] Zhe Jia and Peter Van Sandt. Dissecting the Ampere GPU architecture via microbenchmarking. GPU Technology Conference, 2021.
-   Jia et al. \[2018\] Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. Dissecting the nvidia Volta GPU architecture via microbenchmarking.
    *arXiv preprint arXiv:1804.06826*, 2018.
-   Jia et al. \[2019\] Zhe Jia, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. Dissecting the graphcore IPU architecture via microbenchmarking.
    *arXiv preprint arXiv:1912.03413*, 2019.
-   Johnson et al. \[2016\] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo
    Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. *Scientific data*, 3(1):1--9, 2016.
-   Jouppi et al. \[2017\] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden,
    Al Borchers, et al. In-datacenter performance analysis of a tensor processing unit. In *Proceedings of the 44th annual international symposium on computer
    architecture*, pages 1--12, 2017.
-   Kailath et al. \[1979\] Thomas Kailath, Sun-Yuan Kung, and Martin Morf. Displacement ranks of matrices and linear equations. *Journal of Mathematical
    Analysis and Applications*, 68(2):395--407, 1979.
-   Katharopoulos et al. \[2020\] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive
    transformers with linear attention. In *International Conference on Machine Learning*, pages 5156--5165. PMLR, 2020.
-   Kitaev et al. \[2020\] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In *The International Conference on Machine
    Learning (ICML)*, 2020.
-   Lan et al. \[2020\] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite BEDRT for self-supervised
    learning of language representations. In *The International Conference on Learning Representations (ICLR)*, 2020.
-   Li et al. \[2020\] Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. The deep
    learning compiler: A comprehensive survey. *IEEE Transactions on Parallel and Distributed Systems*, 32(3):708--727, 2020.
-   Likhosherstov et al. \[2020\] Valerii Likhosherstov, Krzysztof Choromanski, Jared Davis, Xingyou Song, and Adrian Weller. Sub-linear memory: How to make
    performers slim. *arXiv preprint arXiv:2012.11346*, 2020.
-   Lin et al. \[2017\] Ji Lin, Yongming Rao, Jiwen Lu, and Jie Zhou. Runtime neural pruning. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
    S. Vishwanathan, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017.
-   Liu et al. \[2019\] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.
    Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.
-   Ma et al. \[2021\] Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. Luna: Linear unified nested attention.
    *Advances in Neural Information Processing Systems*, 34, 2021.
-   Mattson et al. \[2020\] Peter Mattson, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson, Hanlin Tang, Gu-Yeon Wei, Peter
    Bailis, Victor Bittorf, et al. Mlperf training benchmark. *Proceedings of Machine Learning and Systems*, 2:336--349, 2020.
-   McSherry et al. \[2015\] Frank McSherry, Michael Isard, and Derek G Murray. Scalability! but at what $\{$COST$\}$? In *15th Workshop on Hot Topics in
    Operating Systems (HotOS XV)*, 2015.
-   Milakov and Gimelshein \[2018\] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. *arXiv preprint arXiv:1805.02867*, 2018.
-   NVIDIA \[2017\] NVIDIA. Nvidia Tesla V100 GPU architecture, 2017.
-   NVIDIA \[2020\] NVIDIA. Nvidia A100 tensor core GPU architecture, 2020.
-   NVIDIA \[2022\] NVIDIA. Nvidia H100 tensor core GPU architecture, 2022.
-   Parker \[1995\] D Stott Parker. Random butterfly transformations with applications in computational linear algebra.  1995.
-   Paszke et al. \[2019\] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein,
    Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019.
-   Rabe and Staats \[2021\] Markus N Rabe and Charles Staats. Self-attention does not need $O\operatorname{}{(n^{2})}$ memory. *arXiv preprint
    arXiv:2112.05682*, 2021.
-   Radford et al. \[2019\] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask
    learners. *OpenAI blog*, 1(8):9, 2019.
-   Rae and Razavi \[2020\] Jack Rae and Ali Razavi. Do transformers need deep long-range memory? In *Proceedings of the 58th Annual Meeting of the Association
    for Computational Linguistics*, Online, July 2020. Association for Computational Linguistics. URL
    [https://www.aclweb.org/anthology/2020.acl-main.672](https://www.aclweb.org/anthology/2020.acl-main.672){.ltx_ref .ltx_url .ltx_font_typewriter
    target="_blank"}.
-   Rae et al. \[2020\] Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. Compressive transformers for long-range sequence modelling.
    In *The International Conference on Learning Representations (ICLR)*, 2020.
-   Ragan-Kelley et al. \[2013\] Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, and Saman Amarasinghe. Halide: a language
    and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. *Acm Sigplan Notices*, 48(6):519--530, 2013.
-   Ramakrishnan et al. \[2003\] Raghu Ramakrishnan, Johannes Gehrke, and Johannes Gehrke. *Database management systems*, volume 3. McGraw-Hill New York, 2003.
-   Recht and Ré \[2013\] Benjamin Recht and Christopher Ré. Parallel stochastic gradient algorithms for large-scale matrix completion. *Mathematical
    Programming Computation*, 5(2):201--226, 2013.
-   Ren et al. \[2021\] Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale Schuurmans, and Bo Dai. Combiner: Full attention transformer with
    sparse computation cost. *Advances in Neural Information Processing Systems*, 34, 2021.
-   Roy et al. \[2021\] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers.
    *Transactions of the Association for Computational Linguistics*, 9:53--68, 2021.
-   Sabne \[2020\] Amit Sabne. XLA: Compiling machine learning for peak performance.  2020.
-   Sanh et al. \[2020\] Victor Sanh, Thomas Wolf, and Alexander M Rush. Movement pruning: Adaptive sparsity by fine-tuning. *arXiv preprint
    arXiv:2005.07683*, 2020.
-   Shoeybi et al. \[2019\] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training
    multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*, 2019.
-   Sindhwani et al. \[2015\] Vikas Sindhwani, Tara Sainath, and Sanjiv Kumar. Structured transforms for small-footprint deep learning. In *Advances in Neural
    Information Processing Systems*, pages 3088--3096, 2015.
-   Sukhbaatar et al. \[2019\] Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span in transformers. In
    *Proceedings of the Annual Meeting of the Association for Computational Linguistics*, 2019.
-   Tay et al. \[2020a\] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald
    Metzler. Long range arena: A benchmark for efficient transformers. In *International Conference on Learning Representations*, 2020a.
-   Tay et al. \[2020b\] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey. *arXiv preprint arXiv:2009.06732*, 2020b.
-   Vaswani et al. \[2017\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
    Attention is all you need. *Advances in neural information processing systems*, 30, 2017.
-   Wang et al. \[2022\] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. *arXiv
    preprint arXiv:2203.00555*, 2022.
-   Wang et al. \[2020\] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. *arXiv preprint
    arXiv:2006.04768*, 2020.
-   Williams et al. \[2009\] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance model for multicore
    architectures. *Communications of the ACM*, 52(4):65--76, 2009.
-   Wolf and Lam \[1991\] Michael E Wolf and Monica S Lam. A data locality optimizing algorithm. In *Proceedings of the ACM SIGPLAN 1991 conference on
    Programming language design and implementation*, pages 30--44, 1991.
-   Wolf et al. \[2020\] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan
    Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,
    Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical
    Methods in Natural Language Processing: System Demonstrations*, pages 38--45, Online, October 2020. Association for Computational Linguistics. URL
    [https://www.aclweb.org/anthology/2020.emnlp-demos.6](https://www.aclweb.org/anthology/2020.emnlp-demos.6){.ltx_ref .ltx_url .ltx_font_typewriter
    target="_blank"}.
-   Woodruff \[2004\] David P Woodruff. Optimal space lower bounds for all frequency moments. In *SODA*, volume 4, pages 167--175. Citeseer, 2004.
-   Wu et al. \[2019\] Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin, and Michael Auli. Pay less attention with lightweight and dynamic convolutions. In
    *The International Conference on Learning Representations (ICLR)*, 2019.
-   Xiong et al. \[2021\] Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nyströmformer: A nystöm-based
    algorithm for approximating self-attention. In *Proceedings of the AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence*,
    volume 35, page 14138, 2021.
-   Yuan et al. \[2021\] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token
    vit: Training vision transformers from scratch on imagenet. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages
    558--567, 2021.
-   Zaheer et al. \[2020\] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula,
    Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. *Advances in Neural Information Processing Systems*, 33, 2020.
-   Zhai et al. \[2021\] Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, and Josh Susskind. An attention free
    transformer. *arXiv preprint arXiv:2105.14103*, 2021.
-   Zhu et al. \[2021\] Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and Bryan Catanzaro. Long-short transformer:
    Efficient transformers for language and vision. *Advances in Neural Information Processing Systems*, 34, 2021.

## Appendix A Related Work

IO-Aware Runtime Optimization. The broad concept of optimizing for reading and writing to fast/slow memory has a long history in computer science and has been
known by many names. We draw the most direct connection to the literature of analyzing I/O complexity in this work \[[1](#bib.bib1){.ltx_ref}\], but concepts of
memory hierarchies are fundamental and has appeared in many forms, from the working set model \[[21](#bib.bib21){.ltx_ref}\], to data
locality \[[86](#bib.bib86){.ltx_ref}\], to the Roofline model of arithmetic intensity \[[85](#bib.bib85){.ltx_ref}\], to analyses of
scalability \[[59](#bib.bib59){.ltx_ref}\], to standard textbook treatments of computer architecture \[[40](#bib.bib40){.ltx_ref}\]. We hope that this work
encourages the community to adopt these ideas in more parts of the deep learning stack.

Efficient ML Models with Structured Matrices. Matrix multiply is the core computational bottleneck of most machine learning models. To reduce the computational
complexity, there have been numerous approaches to learn over a more efficient set of matrices. These matrices are called *structured matrices*, which have
subquadratic ($o\operatorname{}{(n^{2})}$ for dimension $n \times n$) number of parameters and runtime. Most common examples of structured matrices are sparse
and low-rank matrices, along with fast transforms commonly encountered in signal processing (Fourier, Chebyshev, sine/cosine, orthogonal polynomials). There
have been several more general classes of structured matrices proposed in machine learning: Toeplitz-like \[[78](#bib.bib78){.ltx_ref}\], low-displacement
rank \[[49](#bib.bib49){.ltx_ref}\], quasi-separable \[[25](#bib.bib25){.ltx_ref}\]). The butterfly pattern we use for our block-sparse attention is motivated
by the fact that butterfly matrices \[[64](#bib.bib64){.ltx_ref}, [15](#bib.bib15){.ltx_ref}\] and their products have been shown to be able to express any
structured matrices with almost optimal runtime and number of parameters \[[20](#bib.bib20){.ltx_ref}, [16](#bib.bib16){.ltx_ref}\]. However, even though
structured matrices are efficient in theory, they have not seen wide adoption since it is hard to translate their efficiency to wall-clock speedup since dense
unconstrained matrix multiply has very optimize implementation, a phenomenon known as the hardware lottery \[[41](#bib.bib41){.ltx_ref}\]. Extensions of
butterfly matrices \[[17](#bib.bib17){.ltx_ref}, [18](#bib.bib18){.ltx_ref}\] aimed to make butterfly matrices more hardware-friendly.

Sparse Training. Our block-sparse FlashAttention can be seen as a step towards making sparse model training more efficient. Sparse models have seen success in
compressing models for inference (pruning) by sparsifying the weight matrices \[[39](#bib.bib39){.ltx_ref}, [38](#bib.bib38){.ltx_ref},
[76](#bib.bib76){.ltx_ref}, [55](#bib.bib55){.ltx_ref}, [23](#bib.bib23){.ltx_ref}\]. For model training, the lottery tickets
hypothesis \[[28](#bib.bib28){.ltx_ref}, [29](#bib.bib29){.ltx_ref}, [30](#bib.bib30){.ltx_ref}\] suggests that there are a set of small sub-networks derived
from a larger dense network that performs as well as the original dense network. Out block-sparse FlashAttention can also be seen as a fixed lottery ticket in
the context of attention: we fix the sparsity pattern to be the butterfly pattern through training, and observe that it performs almost as well as the (dense)
FlashAttention on the Long-range Arena tasks.

Efficient Transformer. Transformer-based models have become the most widely-used architecture in natural language processing \[[22](#bib.bib22){.ltx_ref}\] and
computer vision \[[24](#bib.bib24){.ltx_ref}, [91](#bib.bib91){.ltx_ref}\]. However, one of their computational bottlenecks is that their time and memory scales
quadratic in the sequence length. There are numerous approaches to overcome this bottleneck, including approximation with hashing (i.e., sparse) such as
Reformer \[[51](#bib.bib51){.ltx_ref}\] and Smyrf \[[19](#bib.bib19){.ltx_ref}\] and with low-rank approximation such as Performer \[[12](#bib.bib12){.ltx_ref},
[54](#bib.bib54){.ltx_ref}\]. One can even combine sparse and low-rank approximation for better accuracy (e.g., Longformer \[[3](#bib.bib3){.ltx_ref}\],
BigBird \[[92](#bib.bib92){.ltx_ref}\], Scatterbrain \[[9](#bib.bib9){.ltx_ref}\], Long-short transformer \[[94](#bib.bib94){.ltx_ref}\],
Combiner \[[73](#bib.bib73){.ltx_ref}\]). Other approaches include compressing along the sequence dimension to attend to multiple tokens at
once \[[89](#bib.bib89){.ltx_ref}, [79](#bib.bib79){.ltx_ref}, [52](#bib.bib52){.ltx_ref}, [57](#bib.bib57){.ltx_ref}\]. One can also attend over the states
from previous sequences to help lengthen the context (e.g., Transformer-XL \[[14](#bib.bib14){.ltx_ref}\] and Compressive
Transformer \[[69](#bib.bib69){.ltx_ref}\]). We recommend the survey \[[81](#bib.bib81){.ltx_ref}\] for more details.

There are several lines of work on developing other modules instead of attention to model longer context. HiPPO \[[35](#bib.bib35){.ltx_ref}\] and its
extensions, most notably S4 \[[36](#bib.bib36){.ltx_ref}, [37](#bib.bib37){.ltx_ref}, [31](#bib.bib31){.ltx_ref}\] projects the history on a polynomial basis,
allowing accurate reconstruction of the history through state-space models. They combine the strengths of CNNs (efficient training), RNNs (efficient inference),
and continuous models (robust to change in sampling rates). LambdaNetworks \[[2](#bib.bib2){.ltx_ref}\], AFT \[[93](#bib.bib93){.ltx_ref}\] and
FLASH \[[42](#bib.bib42){.ltx_ref}\] are other attempts at replacing attention in the context of image classification and language modeling.

## Appendix B Algorithm Details

We first derive the forward and backward passes of attention and show that they can be computed in a memory-efficient manner (requiring extra memory linear
instead of quadratic in the sequence length). Though they reduce the amount of extra memory required, naively they still incur quadratic HBM accesses, resulting
in slower execution speed. We describe the FlashAttention algorithm to implement both the forward and the backward passes on GPUs that reduces HBM accesses,
leading to both faster runtime and smaller memory footprint.

### B.1 Memory-efficient forward pass

The main challenge in making attention memory-efficient is the softmax that couples the columns of $\mathbf{K}$ (and columns of $\mathbf{V}$). Our approach is
to compute the softmax normalization constant separately to decouple the columns. This technique \[[60](#bib.bib60){.ltx_ref}\] has been used in the
literature \[[51](#bib.bib51){.ltx_ref}, [66](#bib.bib66){.ltx_ref}\] to show that attention computation does not need quadratic *extra* memory (though the
number of HBM accesses is still quadratic, resulting in slow run-time).

For simplicity, we omit here the max-shifting step during softmax. The full algorithm
in [Section B.3](#A2.SS3 "B.3 FlashAttention: Forward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
contains all the steps.

Recall that given input sequences ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$, we want to compute the attention output
$\mathbf{O} \in {\mathbb{R}}^{N \times d}$:

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{\mathbf{S} = {\mathbf{Q}\mathbf{K}}^{\top} \in {\mathbb{R}}^{N \times N}},{\mathbf{P} = {{softmax}\operatorname{}{(\mathbf{S})}} \in {\mathbb{R}}^{N \times N}},{\mathbf{O} = {\mathbf{P}\mathbf{V}} \in {\mathbb{R}}^{N \times d}}}.$$   
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

We have that $S_{i\operatorname{}j} = {q_{i}^{T}\operatorname{}k_{j}}$ where $q_{i}$ and $k_{j}$ are the $i$-th and $j$-th columns of $\mathbf{Q}$ and
$\mathbf{K}$ respectively. Define the normalization constants of softmax:

  -- ------------------------------------------------------------------- -- -----
     $${L_{i} = {\sum\limits_{j}e^{q_{i}^{T}\operatorname{}k_{j}}}}.$$      (1)
  -- ------------------------------------------------------------------- -- -----

Let $v_{j}$ be the $j$-th column of $\mathbf{V}$, then the $i$-th columns of the output is

  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     $${o_{i} = {P_{i:}\operatorname{}\mathbf{V}} = {\sum\limits_{j}{P_{i\operatorname{}j}\operatorname{}v_{j}}} = {\sum\limits_{j}{\frac{e^{q_{i}^{T}\operatorname{}k_{j}}}{L_{i}}\operatorname{}v_{j}}}}.$$      (2)
  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

We see that once $L_{i}$ is computed, we can compute $o_{i}$ without extra memory by repeatedly summing
$\frac{e^{q_{i}^{T}\operatorname{}k_{j}}}{L_{i}}\operatorname{}v_{j}$. Therefore the forward pass can be computed with $O\operatorname{}{(n)}$ extra memory:

1.  1.
    Compute $L_{i}$ for all $i$ according to
    [Eq. 1](#A2.E1 "1 ‣ B.1 Memory-efficient forward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
    which takes $O\operatorname{}{(n)}$ extra memory.
2.  2.
    Compute $o_{i}$ for all $i$ according to
    [Eq. 2](#A2.E2 "2 ‣ B.1 Memory-efficient forward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
    which takes $O\operatorname{}{(d)}$ extra memory.

### B.2 Memory-efficient backward pass

We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe and Staats \[[66](#bib.bib66){.ltx_ref}\] suggests that
the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the
backward pass explicitly and show how it can be computed in a memory-efficient manner.

Suppose that there is a scalar loss function $\phi$, and let the output gradient be ${\mathbf{d}\mathbf{O}} \in {\mathbb{R}}^{n \times d}$ (where
$\mathbf{d}\mathbf{O}$ denotes $\frac{\partial\phi}{\partial\mathbf{O}}$). We want to compute the input gradients
${{\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}} \in {\mathbb{R}}^{n \times d}$ (where
${\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}$ denote
$\frac{\partial\phi}{\partial\mathbf{Q}},\frac{\partial\phi}{\partial\mathbf{K}},\frac{\partial\phi}{\partial\mathbf{V}}$ respectively).

The gradient $\mathbf{d}\mathbf{V}$ is easy to see. Applying reverse-mode autodiff by hand (aka the chain rule), we obtain (in matrix notation)
${\mathbf{d}\mathbf{V}} = {\mathbf{P}^{T}\operatorname{}{\mathbf{d}\mathbf{O}}}$. Thus:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----
     $${{d\operatorname{}v_{j}} = {\sum\limits_{i}{P_{i\operatorname{}j}\operatorname{}d\operatorname{}o_{i}}} = {\sum\limits_{i}{\frac{e^{q_{i}^{T}\operatorname{}k_{j}}}{L_{i}}\operatorname{}d\operatorname{}o_{i}}}}.$$      (3)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----

Since we already computed $L_{i}$, $d\operatorname{}v_{j}$ can be computed without extra memory by repeated summing.

The gradients $\mathbf{d}\mathbf{Q}$ and $\mathbf{d}\mathbf{K}$ are a little more complicated. We go through the gradients $\mathbf{d}\mathbf{P}$ and
$\mathbf{d}\mathbf{S}$ first. From
[Eq. 2](#A2.E2 "2 ‣ B.1 Memory-efficient forward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
we have that ${\mathbf{d}\mathbf{P}} = {\mathbf{d}\mathbf{O}\mathbf{V}}^{T}$, and so:

  -- -------------------------------------------------------------------------------------------------- --
     $${{d\operatorname{}P_{i\operatorname{}j}} = {d\operatorname{}o_{i}^{T}\operatorname{}v_{j}}}.$$   
  -- -------------------------------------------------------------------------------------------------- --

Recall that $P_{i:} = {{softmax}\operatorname{}{(S_{i:})}}$. Using the fact that the Jacobian of $y = {{softmax}\operatorname{}{(x)}}$ is
${{diag}\operatorname{}{(y)}} - {y\operatorname{}y^{T}}$, we have that

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{d\operatorname{}S_{i:}} = {{({{{diag}\operatorname{}{(P_{i:})}} - {P_{i:}\operatorname{}P_{i:}^{T}}})}\operatorname{}d\operatorname{}P_{i:}} = {{{P_{i:} \circ d}\operatorname{}P_{i:}} - {{({P_{i:}^{T}\operatorname{}d\operatorname{}P_{i:}})}\operatorname{}P_{i:}}}},$$   
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

where $\circ$ denotes pointwise multiplication.

Define

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     $${D_{i} = {P_{i:}^{T}\operatorname{}d\operatorname{}P_{i:}} = {\sum\limits_{j}{\frac{e^{q_{i}^{T}\operatorname{}k_{j}}}{L_{i}}\operatorname{}d\operatorname{}o_{i}^{T}\operatorname{}v_{j}}} = {d\operatorname{}o_{i}^{T}\operatorname{}{\sum\limits_{j}{\frac{e^{q_{i}^{\top}\operatorname{}k_{j}}}{L_{i}}\operatorname{}v_{j}}}} = {d\operatorname{}o_{i}^{T}\operatorname{}o_{i}}},$$      (4)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

then

  -- ------------------------------------------------------------------------------------------------------------ --
     $${{d\operatorname{}S_{i:}} = {{{P_{i:} \circ d}\operatorname{}P_{i:}} - {D_{i}\operatorname{}P_{i:}}}}.$$   
  -- ------------------------------------------------------------------------------------------------------------ --

Hence

  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{d\operatorname{}S_{i\operatorname{}j}} = {{P_{i\operatorname{}j}\operatorname{}d\operatorname{}P_{i\operatorname{}j}} - {D_{i}\operatorname{}P_{i\operatorname{}j}}} = {P_{i\operatorname{}j}\operatorname{}{({{d\operatorname{}P_{i\operatorname{}j}} - D_{i}})}}}.$$   
  -- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

Now we can get the gradients $\mathbf{d}\mathbf{Q}$ and $\mathbf{d}\mathbf{K}$. Recall that $S_{i\operatorname{}j} = {q_{i}^{T}\operatorname{}k_{j}}$, so

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     $${{d\operatorname{}q_{i}} = {\sum\limits_{j}{d\operatorname{}S_{i\operatorname{}j}\operatorname{}k_{j}}} = {\sum\limits_{j}{P_{i\operatorname{}j}\operatorname{}{({{d\operatorname{}P_{i\operatorname{}j}} - D_{i}})}\operatorname{}k_{j}}} = {\sum\limits_{j}{\frac{e^{q_{i}^{T}\operatorname{}k_{j}}}{L_{i}}\operatorname{}{({{d\operatorname{}o_{i}^{T}\operatorname{}v_{j}} - D_{i}})}\operatorname{}k_{j}}}}.$$      (5)
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

Similarly,

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     $${{d\operatorname{}k_{j}} = {\sum\limits_{i}{d\operatorname{}S_{i\operatorname{}j}\operatorname{}q_{i}}} = {\sum\limits_{i}{P_{i\operatorname{}j}\operatorname{}{({{d\operatorname{}P_{i\operatorname{}j}} - D_{i}})}\operatorname{}q_{i}}} = {\sum\limits_{i}{\frac{e^{q_{i}^{T}\operatorname{}k_{j}}}{L_{i}}\operatorname{}{({{d\operatorname{}o_{i}^{T}\operatorname{}v_{j}} - D_{i}})}\operatorname{}q_{i}}}}.$$      (6)
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

Therefore the backward pass can also be computed with $O\operatorname{}{(n)}$ extra memory:

1.  1.
    Compute $d\operatorname{}v_{j}$ for all $j$ according to
    [Eq. 3](#A2.E3 "3 ‣ B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
    which takes $O\operatorname{}{(d)}$ extra memory.
2.  2.
    Compute $D_{i}$ for all $i$ according to
    [Eq. 4](#A2.E4 "4 ‣ B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
    which takes $O\operatorname{}{(n)}$ extra memory.
3.  3.
    Compute $d\operatorname{}q_{i}$ for all $i$ according to
    [Eq. 5](#A2.E5 "5 ‣ B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
    which takes $O\operatorname{}{(d)}$ extra memory.
4.  4.
    Compute $d\operatorname{}k_{j}$ for all $j$ according to
    [Eq. 6](#A2.E6 "6 ‣ B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
    which takes $O\operatorname{}{(d)}$ extra memory.

### B.3 FlashAttention: Forward Pass

We describe the full details of FlashAttention forward pass. Given input sequences ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$, we want
to compute the attention output $\mathbf{O} \in {\mathbb{R}}^{N \times d}$:

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     ${{\mathbf{S} = {\tau\operatorname{}{\mathbf{Q}\mathbf{K}}^{\top}} \in {\mathbb{R}}^{N \times N}},{\mathbf{S}^{masked} = {\text{mask}\operatorname{}{(S)}} \in {\mathbb{R}}^{N \times N}},{\mathbf{P} = {{softmax}\operatorname{}{(\mathbf{S}^{masked})}} \in {\mathbb{R}}^{N \times N}}},$   
     ${{\mathbf{P}^{dropped} = {{dropout}\operatorname{}{(\mathbf{P},p_{drop})}}},{\mathbf{O} = {\mathbf{P}^{dropped}\operatorname{}\mathbf{V}} \in {\mathbb{R}}^{N \times d}}},$                                                                                                                  
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

where $\tau \in {\mathbb{R}}$ is some softmax scaling (typically $\frac{1}{\sqrt{d}}$), mask is some masking function that sets some entries of the input to
$- \infty$ and keep other entries the same (e.g., key padding mask when sequences in the batch don't have the same lengths and are padded), and
${dropout}\operatorname{}{(x,p)}$ applies dropout to $x$ elementwise (i.e., output $\frac{x}{1 - p}$ with probability $1 - p$ and output 0 with probability $p$
for each element $x$).

The full algorithm is
in [Algorithm 2](#alg2 "Algorithm 2 ‣ B.3 FlashAttention: Forward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
We save the output $\mathbf{O}$, the softmax statistics $\ell$ and $m$, and the pseudo-random number generator state $\mathcal{R}$ for the backward pass.

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ in HBM, on-chip SRAM of size $M$, softmax scaling constant
$\tau \in {\mathbb{R}}$, masking function mask, dropout probability $p_{drop}$.

1:  Initialize the pseudo-random number generator state $\mathcal{R}$ and save to HBM.

2:  Set block sizes
${B_{c} = \left\lceil \frac{M}{4\operatorname{}d} \right\rceil},{B_{r} = {\min\left( \left\lceil \frac{M}{4\operatorname{}d} \right\rceil,d \right)}}$.

3:  Initialize
${\mathbf{O} = {(0)}_{N \times d} \in {\mathbb{R}}^{N \times d}},{\ell = {(0)}_{N} \in {\mathbb{R}}^{N}},{m = {({- \infty})}_{N} \in {\mathbb{R}}^{N}}$ in HBM.

4:  Divide $\mathbf{Q}$ into $T_{r} = \left\lceil \frac{N}{B_{r}} \right\rceil$ blocks $\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}$ of size $B_{r} \times d$ each,
and divide $\mathbf{K},\mathbf{V}$ in to $T_{c} = \left\lceil \frac{N}{B_{c}} \right\rceil$ blocks $\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}$ and
$\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}$, of size $B_{c} \times d$ each.

5:  Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}$ of size $B_{r} \times d$ each, divide $\ell$ into $T_{r}$ blocks
$\ell_{i},\ldots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\ldots,m_{T_{r}}$ of size $B_{r}$ each.

6:  for $1 \leq j \leq T_{c}$ do

7:     Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.

8:     for $1 \leq i \leq T_{r}$ do

9:        Load $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.

10:        On chip, compute
$\mathbf{S}_{i\operatorname{}j} = {\tau\operatorname{}\mathbf{Q}_{i}\operatorname{}\mathbf{K}_{j}^{T}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

11:        On chip, compute $\mathbf{S}_{i\operatorname{}j}^{masked} = {\text{mask}\operatorname{}{(\mathbf{S}_{i\operatorname{}j})}}$.

12:        On chip, compute
${\overset{\sim}{m}}_{i\operatorname{}j} = {{rowmax}\operatorname{}{(\mathbf{S}_{i\operatorname{}j}^{masked})}} \in {\mathbb{R}}^{B_{r}}$,
${\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j} = {\exp{({\mathbf{S}_{i\operatorname{}j}^{masked} - {\overset{\sim}{m}}_{i\operatorname{}j}})}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$
(pointwise),
${\overset{\sim}{\ell}}_{i\operatorname{}j} = {{rowsum}\operatorname{}{({\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j})}} \in {\mathbb{R}}^{B_{r}}$.

13:        On chip, compute $m_{i}^{new} = {\max{(m_{i},{\overset{\sim}{m}}_{i\operatorname{}j})}} \in {\mathbb{R}}^{B_{r}}$,
$\ell_{i}^{new} = {{e^{m_{i} - m_{i}^{new}}\operatorname{}\ell_{i}} + {e^{{\overset{\sim}{m}}_{i\operatorname{}j} - m_{i}^{new}}\operatorname{}{\overset{\sim}{\ell}}_{i\operatorname{}j}}} \in {\mathbb{R}}^{B_{r}}$.

14:        On chip, compute
${\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j}^{dropped} = {{dropout}\operatorname{}{({\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j},p_{drop})}}$.

15:        Write
$\mathbf{O}_{i}\leftarrow{{diag}\operatorname{}{(\ell_{i}^{new})}^{- 1}\operatorname{}{({{{diag}\operatorname{}{(\ell_{i})}\operatorname{}e^{m_{i} - m_{i}^{new}}\operatorname{}\mathbf{O}_{i}} + {e^{{\overset{\sim}{m}}_{i\operatorname{}j} - m_{i}^{new}}\operatorname{}{\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j}^{dropped}\operatorname{}\mathbf{V}_{j}}})}}$
to HBM.

16:        Write $\ell_{i}\leftarrow\ell_{i}^{new}$, $m_{i}\leftarrow m_{i}^{new}$ to HBM.

17:     end for

18:  end for

19:  Return $\mathbf{O},\ell,m,\mathcal{R}$.

Algorithm 2 FlashAttention Forward Pass

### B.4 FlashAttention: Backward Pass

We describe the full details of FlashAttention backward pass. Given input sequences ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$, the
output $\mathbf{O} \in {\mathbb{R}}^{N \times d}$, and the output gradient $\mathbf{d}\mathbf{O}$, we want to compute the input gradients
${{\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}} \in {\mathbb{R}}^{N \times d}$.

We first describe the standard attention backward pass
in [Algorithm 3](#alg3 "Algorithm 3 ‣ B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
for completeness.

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V},{\mathbf{d}\mathbf{O}}} \in {\mathbb{R}}^{N \times d}$, $\mathbf{P} \in {\mathbb{R}}^{N \times N}$ in HBM.

1:  Load $\mathbf{P},{\mathbf{d}\mathbf{O}}$ by blocks from HBM, compute
${\mathbf{d}\mathbf{V}} = {\mathbf{P}^{\top}\operatorname{}{\mathbf{d}\mathbf{O}}} \in {\mathbb{R}}^{N \times d}$, write $\mathbf{d}\mathbf{V}$ to HBM.

2:  Load ${\mathbf{d}\mathbf{O}},\mathbf{V}$ by blocks from HBM, compute
${\mathbf{d}\mathbf{P}} = {\mathbf{d}\mathbf{O}\mathbf{V}}^{\top} \in {\mathbb{R}}^{N \times N}$, write $\mathbf{d}\mathbf{P}$ to HBM.

3:  Read $\mathbf{P},{\mathbf{d}\mathbf{P}}$ from HBM, compute ${\mathbf{d}\mathbf{S}} \in {\mathbb{R}}^{N \times N}$ where
${d\operatorname{}S_{i\operatorname{}j}} = {P_{i\operatorname{}j}\operatorname{}{({{d\operatorname{}P_{i\operatorname{}j}} - {\sum_{l}{P_{i\operatorname{}l}\operatorname{}d\operatorname{}P_{i\operatorname{}l}}}})}}$,
write $\mathbf{d}\mathbf{S}$ to HBM.

4:  Load $\mathbf{d}\mathbf{S}$ and $\mathbf{K}$ by blocks from HBM, compute ${\mathbf{d}\mathbf{Q}} = {\mathbf{d}\mathbf{S}\mathbf{K}}$, write
$\mathbf{d}\mathbf{Q}$ to HBM.

5:  Load $\mathbf{d}\mathbf{S}$ and $\mathbf{Q}$ by blocks from HBM, compute
${\mathbf{d}\mathbf{K}} = {{\mathbf{d}\mathbf{S}}^{\top}\operatorname{}\mathbf{Q}}$, write $\mathbf{d}\mathbf{K}$ to HBM.

6:  Return ${\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}$.

Algorithm 3 Standard Attention Backward Pass

We now make two observations about FlashAttention backward pass:

1.  1.
    We do not need to store the dropout mask of size $O\operatorname{}{(N^{2})}$ from the forward pass. Instead, we can save the pseudo-random number generator
    states from the forward pass and re-generate the dropout mask in the backward pass. This allows us to only use $O\operatorname{}{(N)}$ extra memory.
2.  2.
    When computing the softmax gradient, we
    use [Eq. 4](#A2.E4 "4 ‣ B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
    to compute $D_{i} = {P_{i:}^{\top}\operatorname{}d\operatorname{}P_{i:}}$ without reducing over $P_{i:}$ and $d\operatorname{}P_{i:}$ of size $N$ (they
    might not fit into SRAM). Instead we can rewrite $D_{i} = {d\operatorname{}o_{i}^{\top}\operatorname{}o_{i}}$ and compute the dot product between vectors of
    size $d$.

The full FlashAttention backward pass algorithm is
in [Algorithm 4](#alg4 "Algorithm 4 ‣ B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
Conceptually it is just a block version of the derivation
in [Section B.2](#A2.SS2 "B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},{\mathbf{d}\mathbf{O}}} \in {\mathbb{R}}^{N \times d}$ in HBM, vectors
${\ell,m} \in {\mathbb{R}}^{N}$ in HBM, on-chip SRAM of size $M$, softmax scaling constant $\tau \in {\mathbb{R}}$, masking function mask, dropout probability
$p_{drop}$, pseudo-random number generator state $\mathcal{R}$ from the forward pass.

1:  Set the pseudo-random number generator state to $\mathcal{R}$.

2:  Set block sizes
${B_{c} = \left\lceil \frac{M}{4\operatorname{}d} \right\rceil},{B_{r} = {\min\left( \left\lceil \frac{M}{4\operatorname{}d} \right\rceil,d \right)}}$.

3:  Divide $\mathbf{Q}$ into $T_{r} = \left\lceil \frac{N}{B_{r}} \right\rceil$ blocks $\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}$ of size $B_{r} \times d$ each,
and divide $\mathbf{K},\mathbf{V}$ in to $T_{c} = \left\lceil \frac{N}{B_{c}} \right\rceil$ blocks $\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}$ and
$\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}$, of size $B_{c} \times d$ each.

4:  Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}$ of size $B_{r} \times d$ each, divide $\mathbf{d}\mathbf{O}$ into $T_{r}$
blocks ${\mathbf{d}\mathbf{O}}_{i},\ldots,{\mathbf{d}\mathbf{O}}_{T_{r}}$ of size $B_{r} \times d$ each, divide $\ell$ into $T_{r}$ blocks
$\ell_{i},\ldots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\ldots,m_{T_{r}}$ of size $B_{r}$ each.

5:  Initialize ${\mathbf{d}\mathbf{Q}} = {(0)}_{N \times d}$ in HBM and divide it into $T_{r}$ blocks
${\mathbf{d}\mathbf{Q}}_{1},\ldots,{\mathbf{d}\mathbf{Q}}_{T_{r}}$ of size $B_{r} \times d$ each. Initialize
${{\mathbf{d}\mathbf{K}} = {(0)}_{N \times d}},{{\mathbf{d}\mathbf{V}} = {(0)}_{N \times d}}$ in HBM and divide ${\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}$
in to $T_{c}$ blocks ${\mathbf{d}\mathbf{K}}_{1},\ldots,{\mathbf{d}\mathbf{K}}_{T_{c}}$ and ${\mathbf{d}\mathbf{V}}_{1},\ldots,{\mathbf{d}\mathbf{V}}_{T_{c}}$,
of size $B_{c} \times d$ each.

6:  for $1 \leq j \leq T_{c}$ do

7:     Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.

8:     Initialize ${{\overset{\sim}{\mathbf{d}\mathbf{K}}}_{j} = {(0)}_{B_{c} \times d}},{{\overset{\sim}{\mathbf{d}\mathbf{V}}}_{j} = {(0)}_{B_{c} \times d}}$
on SRAM.

9:     for $1 \leq i \leq T_{r}$ do

10:        Load $\mathbf{Q}_{i},\mathbf{O}_{i},{\mathbf{d}\mathbf{O}}_{i},{\mathbf{d}\mathbf{Q}}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.

11:        On chip, compute
$\mathbf{S}_{i\operatorname{}j} = {\tau\operatorname{}\mathbf{Q}_{i}\operatorname{}\mathbf{K}_{j}^{T}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

12:        On chip, compute $\mathbf{S}_{i\operatorname{}j}^{masked} = {\text{mask}\operatorname{}{(\mathbf{S}_{i\operatorname{}j})}}$.

13:        On chip, compute
$\mathbf{P}_{i\operatorname{}j} = {{diag}\operatorname{}{(l_{i})}^{- 1}\operatorname{}{\exp{({\mathbf{S}_{i\operatorname{}j}^{masked} - m_{i}})}}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

14:        On chip, compute dropout mask $\mathbf{Z}_{i\operatorname{}j} \in {\mathbb{R}}^{B_{r} \times B_{c}}$ where each entry has value
$\frac{1}{1 - p_{drop}}$ with probability $1 - p_{drop}$ and value 0 with probability $p_{drop}$.

15:        On chip, compute $\mathbf{P}_{i\operatorname{}j}^{dropped} = {\mathbf{P}_{i\operatorname{}j} \circ \mathbf{Z}_{i\operatorname{}j}}$ (pointwise
multiply).

16:        On chip, compute
$\overset{\sim}{{\mathbf{d}\mathbf{V}}_{j}}\leftarrow{\overset{\sim}{{\mathbf{d}\mathbf{V}}_{j}} + {{(\mathbf{P}_{i\operatorname{}j}^{dropped})}^{\top}\operatorname{}{\mathbf{d}\mathbf{O}}_{i}}} \in {\mathbb{R}}^{B_{c} \times d}$.

17:        On chip, compute
${\mathbf{d}\mathbf{P}}_{i\operatorname{}j}^{dropped} = {{\mathbf{d}\mathbf{O}}_{i}\operatorname{}\mathbf{V}_{j}^{\top}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

18:        On chip, compute
${\mathbf{d}\mathbf{P}}_{i\operatorname{}j} = {{\mathbf{d}\mathbf{P}}_{i\operatorname{}j}^{dropped} \circ \mathbf{Z}_{i\operatorname{}j}}$ (pointwise multiply).

19:        On chip, compute $D_{i} = {{rowsum}\operatorname{}{({{\mathbf{d}\mathbf{O}}_{i} \circ \mathbf{O}_{i}})}} \in {\mathbb{R}}^{B_{r}}$.

20:        On chip, compute
${\mathbf{d}\mathbf{S}}_{i\operatorname{}j} = {\mathbf{P}_{i\operatorname{}j} \circ {({{\mathbf{d}\mathbf{P}}_{i\operatorname{}j} - D_{i}})}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

21:        Write
${\mathbf{d}\mathbf{Q}}_{i}\leftarrow{{\mathbf{d}\mathbf{Q}}_{i} + {\tau\operatorname{}{\mathbf{d}\mathbf{S}}_{i\operatorname{}j}\operatorname{}\mathbf{K}_{j}}} \in {\mathbb{R}}^{B_{r} \times d}$
to HBM.

22:        On chip, compute
${\overset{\sim}{\mathbf{d}\mathbf{K}}}_{j}\leftarrow{{\overset{\sim}{\mathbf{d}\mathbf{K}}}_{j} + {\tau\operatorname{}{\mathbf{d}\mathbf{S}}_{i\operatorname{}j}^{\top}\operatorname{}\mathbf{Q}_{i}}} \in {\mathbb{R}}^{B_{c} \times d}$.

23:     end for

24:     Write
${{\mathbf{d}\mathbf{K}}_{j}\leftarrow\overset{\sim}{{\mathbf{d}\mathbf{K}}_{j}}},{{\mathbf{d}\mathbf{V}}_{j}\leftarrow\overset{\sim}{{\mathbf{d}\mathbf{V}}_{j}}}$
to HBM.

25:  end for

26:  Return ${\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}$.

Algorithm 4 FlashAttention Backward Pass

We see that similar to the forward pass, the backward pass performs $O\operatorname{}{(N^{2})}$ FLOPs and only requires $O\operatorname{}{(N)}$ extra memory
beyond inputs, output, output gradient, and input gradients.

We analyze the IO-complexity of the backward pass, similar to the forward pass
([Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).

###### Theorem 5.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d \leq M \leq {N\operatorname{}d}$. Standard attention
([Algorithm ](#alg0 "Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
backward pass requires $\Theta\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ HBM accesses, while FlashAttention backward pass
([Algorithm 4](#alg4 "Algorithm 4 ‣ B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
requires $\Theta\operatorname{}{({N^{2}\operatorname{}d^{2}\operatorname{}M^{- 1}})}$ HBM accesses.

The proof is in [Appendix C](#A3 "Appendix C Proofs ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

### B.5 Comparison with Rabe and Staats \[[66](#bib.bib66){.ltx_ref}\]

We describe here some similarities and differences between our FlashAttention algorithm and the algorithm of Rabe and Staats \[[66](#bib.bib66){.ltx_ref}\].

Conceptually, both FlashAttention and Rabe and Staats \[[66](#bib.bib66){.ltx_ref}\] operate on blocks of the attention matrix using the well-established
technique of tiling (or softmax scaling) \[[60](#bib.bib60){.ltx_ref}, [51](#bib.bib51){.ltx_ref}\]. To reduce the memory footprint, both methods avoid storing
the large attention matrix in the forward pass and recompute it in the backward pass.

The first major difference is that Rabe and Staats \[[66](#bib.bib66){.ltx_ref}\] focuses on the reducing the total memory footprint (maximum amount of GPU
memory required) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned
in [Section 2](#S2 "2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}, the amount of memory access is the
primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amount of memory required (e.g., if an operation incurs $A$
memory accesses, then its total memory requirement is at most $A$). As a result, FlashAttention is faster than standard attention (2-4$\times$) while Rabe and
Staats \[[66](#bib.bib66){.ltx_ref}\] is around the same speed or slightly slower than standard attention. In terms of total memory required, both methods offer
substantial memory saving.

The second difference between the two methods is the way information is summarized from each block to pass to the next block. Rabe and Staats
\[[66](#bib.bib66){.ltx_ref}\] summarizes each block with its temporary output along with the softmax normalization statistics. At the end of the forward pass,
the temporary outputs of all the blocks are combined using the statistics to produce the final output. FlashAttention instead incrementally updates the output
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[12](#alg1.l12 "12 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
after processing each block, so only one copy of the output is needed (instead of $K$ copies for $K$ blocks). This means that FlashAttention has smaller total
memory requirement compared to Rabe and Staats \[[66](#bib.bib66){.ltx_ref}\].

The final major difference is the way the backward pass is computed. Rabe and Staats \[[66](#bib.bib66){.ltx_ref}\] uses gradient checkpointing to recompute the
attention matrix and the temporary output of each block. FlashAttention instead simplifies the backward pass analytically
([Sections B.2](#A2.SS2 "B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
and [B.4](#A2.SS4 "B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
It only recomputes the attention matrix and does not recompute the temporary output of each block. This reduces the memory requirement for the backward pass and
yields speedup.

## Appendix C Proofs

###### Proof of [Theorem 1](#Thmtheorem1 "Theorem 1. ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

We first count the number of FLOPs and extra memory required.

The dominating FLOPs are from matrix multiplication. In the inner loop,
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[9](#alg1.l9 "9 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}),
we compute ${\mathbf{Q}_{i}\operatorname{}\mathbf{K}_{j}^{\top}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$ for $\mathbf{Q}_{i} \in {\mathbb{R}}^{B_{r} \times d}$
and $\mathbf{K}_{j} \in {\mathbb{R}}^{B_{c} \times d}$, which takes $O\operatorname{}{({B_{r}\operatorname{}B_{c}\operatorname{}d})}$ FLOPs. We also compute
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[12](#alg1.l12 "12 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
${{\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j}\operatorname{}\mathbf{V}_{j}} \in {\mathbb{R}}^{B_{r} \times d}$ for
${\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j} \in {\mathbb{R}}^{B_{r} \times B_{c}}$ and $\mathbf{V}_{j} \in {\mathbb{R}}^{B_{c} \times d}$, which takes
$O\operatorname{}{({B_{r}\operatorname{}B_{c}\operatorname{}d})}$ FLOPs. We execute the inner loops
${T_{c}\operatorname{}T_{r}} = {\left\lceil \frac{N}{B_{c}} \right\rceil\operatorname{}\left\lceil \frac{N}{B_{r}} \right\rceil}$ times. Therefore the total
number of FLOPs is

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{O\operatorname{}\left( {\frac{N^{2}}{B_{c}\operatorname{}B_{r}}\operatorname{}B_{r}\operatorname{}B_{c}\operatorname{}d} \right)} = {O\operatorname{}{({N^{2}\operatorname{}d})}}}.$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

In terms of extra memory required, we see that we need $O\operatorname{}{(N)}$ memory to store the statistics $(\ell,m)$.

We now prove the algorithm's correctness by induction on $j$ for $0 \leq j \leq T_{c}$. Let
$\mathbf{K}_{:j} \in {\mathbb{R}}^{{j\operatorname{}B_{c}} \times d}$ be the first $j\operatorname{}B_{c}$ rows of $\mathbf{K}$, and similarly
$\mathbf{V}_{:j} \in {\mathbb{R}}^{{j\operatorname{}B_{c}} \times d}$ the the first $j\operatorname{}B_{c}$ rows of $\mathbf{V}$. Let
$\mathbf{S}_{:,{:j}} = {\mathbf{Q}\mathbf{K}}_{:j}^{\top} \in {\mathbb{R}}^{{N \times j}\operatorname{}B_{c}}$, and
$\mathbf{P}_{:,{:j}} = {{softmax}\operatorname{}{(\mathbf{S}_{:,{:j}})}} \in {\mathbb{R}}^{{N \times j}\operatorname{}B_{c}}$ (softmax applied row-wise). Let
$m^{j},\ell^{(j)},\mathbf{O}^{(j)}$ be the values of $m,\ell,\mathbf{O}$ in HBM after the $j$-th iteration of the outer loop
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[5](#alg1.l5 "5 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
(Note that these values of $m,\ell,\mathbf{O}$ are updated after each iteration of the outer loop.) We want to show that after the $j$-th iteration of the outer
loop, we have computed in HBM:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --
     $${{m^{(j)} = {{rowmax}\operatorname{}{(\mathbf{S}_{:,{:j}})}} \in {\mathbb{R}}^{N}},{\ell^{(j)} = {{rowsum}\operatorname{}{({\exp{({\mathbf{S}_{:,{:j}} - m^{(j)}})}})}} \in {\mathbb{R}}^{N}},{\mathbf{O}^{(j)} = {\mathbf{P}_{:,{:j}}\operatorname{}\mathbf{V}_{:j}} \in {\mathbb{R}}^{N \times d}}}.$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --

Based on our initialization
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[2](#alg1.l2 "2 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}),
this claim is true for $j = 0$ (i.e., before the any iteration of the outer loop is executed). Suppose that the claim holds for some
$j = {0,\ldots,{T_{c} - 1}}$. We want to show that the claim also holds for $j + 1$. Indeed, when we update the statistics in the inner loop
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line
[10](#alg1.l10 "10 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref})
on the $({j + 1})$-th iteration of the outer loop, we update $m^{({j + 1})} = {\max{(m^{(j)},\overset{\sim}{m})}}$ where
$\overset{\sim}{m} \in {\mathbb{R}}^{N}$ is the row-max of $\mathbf{S}_{{:,j}:{j + 1}}$, the slice of $\mathbf{S}$ from column $j\operatorname{}B_{c}$ to column
${{({j + 1})}\operatorname{}B_{c}} - 1$. This implies that

  -- ---------------------------------------------------------------------------------------------------- --
     $${m^{({j + 1})} = {{rowmax}\operatorname{}{(\mathbf{S}_{:,{:{j + 1}}})}} \in {\mathbb{R}}^{N}}.$$   
  -- ---------------------------------------------------------------------------------------------------- --

Similarly, we update

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${\ell^{({j + 1})} = {{e^{m^{(j)} - m^{({j + 1})}}\operatorname{}\ell^{(j)}} + {e^{\overset{\sim}{m} - m^{({j + 1})}}\operatorname{}\overset{\sim}{\ell}}}},$$   
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------- --

where $\overset{\sim}{\ell} = {{rowsum}\operatorname{}{({\exp{({\mathbf{S}_{{:,j}:{j + 1}} - \overset{\sim}{m}})}})}} \in {\mathbb{R}}^{N}$. By the same
algebraic manipulation
in [Section 3.1](#S3.SS1 "3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
we obtain:

  -- ----------------------------------------------------------------------------------------------------------------------------------- --
     $${\ell^{({j + 1})} = {{rowsum}\operatorname{}{({\exp{({\mathbf{S}_{:,{:{j + 1}}} - m^{({j + 1})}})}})}} \in {\mathbb{R}}^{N}}.$$   
  -- ----------------------------------------------------------------------------------------------------------------------------------- --

Let $\mathbf{V}_{j:{j + 1}}$ be the slice of $\mathbf{V}$ from column $j\operatorname{}B_{c}$ to column ${{({j + 1})}\operatorname{}B_{c}} - 1$, we also update:

  -- -------------------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $\mathbf{O}^{({j + 1})}$   $= {{diag}\operatorname{}{(\ell^{({j + 1})})}^{- 1}\operatorname{}{({{{diag}\operatorname{}{(\ell^{(j)})}\operatorname{}e^{m^{(j)} - m^{({j + 1})}}\operatorname{}\mathbf{O}^{(j)}} + {e^{\overset{\sim}{m} - m^{({j + 1})}}\operatorname{}{\exp{({\mathbf{S}_{j:{j + 1}} - \overset{\sim}{m}})}}\operatorname{}\mathbf{V}_{j:{j + 1}}}})}}$                                                                    
                                $= {{diag}\operatorname{}{(\ell^{({j + 1})})}^{- 1}\operatorname{}{({{{diag}\operatorname{}{(\ell^{(j)})}\operatorname{}e^{m^{(j)} - m^{({j + 1})}}\operatorname{}\mathbf{P}_{:,{:j}}\operatorname{}\mathbf{V}_{:j}} + {e^{- m^{({j + 1})}}\operatorname{}{\exp{(\mathbf{S}_{j:{j + 1}})}}\operatorname{}\mathbf{V}_{j:{j + 1}}}})}}$                                                                           
                                $= {{diag}\operatorname{}{(\ell^{({j + 1})})}^{- 1}\operatorname{}{({{{diag}\operatorname{}{(\ell^{(j)})}\operatorname{}e^{m^{(j)} - m^{({j + 1})}}\operatorname{}{diag}\operatorname{}{(\ell^{(j)})}\operatorname{}{\exp{({\mathbf{S}_{:,{:j}} - m^{(j)}})}}\operatorname{}\mathbf{V}_{:j}} + {e^{- m^{({j + 1})}}\operatorname{}{\exp{(\mathbf{S}_{j:{j + 1}})}}\operatorname{}\mathbf{V}_{j:{j + 1}}}})}}$   
                                $= {{diag}\operatorname{}{(\ell^{({j + 1})})}^{- 1}\operatorname{}{({{e^{- m^{({j + 1})}}\operatorname{}{\exp{(\mathbf{S}_{:,{:j}})}}\operatorname{}\mathbf{V}_{:j}} + {e^{- m^{({j + 1})}}\operatorname{}{\exp{(\mathbf{S}_{j:{j + 1}})}}\operatorname{}\mathbf{V}_{j:{j + 1}}}})}}$                                                                                                                           
                                $= {{diag}\operatorname{}{(\ell^{({j + 1})})}^{- 1}\operatorname{}{({{{\exp{({\mathbf{S}_{:,{:j}} - m^{({j + 1})}})}}\operatorname{}\mathbf{V}_{:j}} + {{\exp{({\mathbf{S}_{j:{j + 1}} - m^{({j + 1})}})}}\operatorname{}\mathbf{V}_{j:{j + 1}}}})}}$                                                                                                                                                           
                                $= {{diag}\operatorname{}{(\ell^{({j + 1})})}^{- 1}\operatorname{}\left( {\exp\left( {\begin{bmatrix}                                                                                                                                                                                                                                                                                                           
                                \mathbf{S}_{:,{:j}} & \mathbf{S}_{j:{j + 1}}                                                                                                                                                                                                                                                                                                                                                                    
                                \end{bmatrix} - m^{({j + 1})}} \right)} \right)\operatorname{}\begin{bmatrix}                                                                                                                                                                                                                                                                                                                                   
                                \mathbf{V}_{:j} \\                                                                                                                                                                                                                                                                                                                                                                                              
                                \mathbf{V}_{j:{j + 1}}                                                                                                                                                                                                                                                                                                                                                                                          
                                \end{bmatrix}}$                                                                                                                                                                                                                                                                                                                                                                                                 
                                ${= {{softmax}\operatorname{}{(\mathbf{S}_{:{j + 1}})}\operatorname{}\mathbf{V}_{:{j + 1}}}}.$                                                                                                                                                                                                                                                                                                                  
  -- -------------------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

We then see that the claim is also true for $j + 1$. By induction, the claim is true for all $j = {0,\ldots,T_{c}}$.

When $j = T_{c}$, we conclude that the final value of $\mathbf{O}$ in HBM is
${{softmax}\operatorname{}{(\mathbf{S})}\operatorname{}\mathbf{V}} = {{softmax}\operatorname{}{({\mathbf{Q}\mathbf{K}}^{\top})}\operatorname{}\mathbf{V}}$.

∎

###### Proof of [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

We first analyze the IO complexity of standard attention implementation. The inputs ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ reside in
HBM, and the at the end of the algorithm the output $\mathbf{O} \in {\mathbb{R}}^{N \times d}$ is written to HBM.

In the first step of computing the matrix multiply $\mathbf{S} = {\mathbf{Q}\mathbf{K}}^{\top}$, the inputs $\mathbf{Q},\mathbf{K}$ are read from HBM and the
output $\mathbf{S} \in {\mathbb{R}}^{N \times N}$ is written to HBM
([Algorithm ](#alg0 "Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line [1](#alg0.l1 "1 ‣ Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
This incurs $\Theta\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ HBM accesses.

In the second step of computing $\mathbf{P} = {{softmax}\operatorname{}{(\mathbf{S})}}$, the input $\mathbf{S}$ is read from HBM and the output $\mathbf{P}$ is
written to HBM
([Algorithm ](#alg0 "Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line [2](#alg0.l2 "2 ‣ Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
This incurs $\Theta\operatorname{}{(N^{2})}$ HBM accesses.

In the last step of computing $\mathbf{O} = {\mathbf{P}\mathbf{V}}$, the inputs $\mathbf{P},\mathbf{V}$ are read from global memory and the output $\mathbf{O}$
is written to HBM
([Algorithm ](#alg0 "Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line [3](#alg0.l3 "3 ‣ Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
This incurs $\Theta\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ HBM accesses.

Overall, standard attention implementation requires $\Theta\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ global memory accesses.

We now analyze the IO complexity of streaming attention.

Following [Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
we see that each element of $\mathbf{K}$ and $\mathbf{V}$ is loaded from HBM once
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line [6](#alg1.l6 "6 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
We make $T_{c}$ passes over $\mathbf{Q}$ and $\mathbf{O}$, each pass loading all of $\mathbf{Q}$ and all of $\mathbf{O}$ to HBM
([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
line [8](#alg1.l8 "8 ‣ Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
Therefore the number of HBM accesses is
${\Theta\operatorname{}\left( {{N\operatorname{}d} + {N\operatorname{}d\operatorname{}T_{c}}} \right)} = {\Theta\operatorname{}{({N\operatorname{}d\operatorname{}T_{c}})}}$.

We derive the conditions on the block sizes $B_{c}$ and $B_{r}$. We need the blocks $\mathbf{K}_{j}$ and $\mathbf{V}_{j}$ of size $B_{c} \times d$ to fit into
on-chip memory, which translates to:

  -- --------------------------------------------------------------------------------------------------------------------------------- --
     $${{{B_{c}\operatorname{}d} = {O\operatorname{}{(M)}}}\Leftrightarrow{B_{c} = {O\operatorname{}\left( \frac{M}{d} \right)}}}.$$   
  -- --------------------------------------------------------------------------------------------------------------------------------- --

Similarly, we need the blocks $\mathbf{Q}_{i},\mathbf{O}_{i}$ of size $B_{r} \times d$ to fit into on-chip memory, which translates to:

  -- --------------------------------------------------------------------------------------------------------------------------------- --
     $${{{B_{r}\operatorname{}d} = {O\operatorname{}{(M)}}}\Leftrightarrow{B_{r} = {O\operatorname{}\left( \frac{M}{d} \right)}}}.$$   
  -- --------------------------------------------------------------------------------------------------------------------------------- --

Finally, we need the block $\mathbf{S}_{i\operatorname{}j}$ of size $B_{r} \times B_{c}$ to fit into on-chip memory, which translates to:

  -- -------------------------------------------------------------- --
     $${{B_{r}\operatorname{}B_{c}} = {O\operatorname{}{(M)}}}.$$   
  -- -------------------------------------------------------------- --

We therefore set:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --
     $${{B_{c} = {\Theta\operatorname{}\left( \frac{M}{d} \right)}},{B_{r} = {\Theta\operatorname{}\left( {\min\left( \frac{M}{d},\frac{M}{B_{c}} \right)} \right)} = {\Theta\operatorname{}\left( {\min\left( \frac{M}{d},d \right)} \right)}}}.$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --

We then have:

  -- ---------------------------------------------------------------------------------------------------- --
     $${T_{c} = \frac{N}{B_{c}} = {\Theta\operatorname{}\left( \frac{N\operatorname{}d}{M} \right)}}.$$   
  -- ---------------------------------------------------------------------------------------------------- --

As a result, the number of HBM accesses is:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------ --
     $${{\Theta\operatorname{}\left( {N\operatorname{}d\operatorname{}T_{c}} \right)} = {\Theta\operatorname{}\left( \frac{N^{2}\operatorname{}d^{2}}{M} \right)}}.$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------ --

∎

###### Proof of [Proposition 3](#Thmtheorem3 "Proposition 3. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

For contradiction, suppose that there exists an algorithm that computes exact attention where the number for HBM access for all
$M \in {\lbrack d,{N\operatorname{}d}\rbrack}$ is

  -- --------------------------------------------------------------------------- --
     $${o\operatorname{}\left( \frac{N^{2}\operatorname{}d^{2}}{M} \right)}.$$   
  -- --------------------------------------------------------------------------- --

In the regime of $M = {\Theta\operatorname{}{({N\operatorname{}d})}}$, this results in the number of HBM accesses:

  -- ----------------------------------------------------------------------------------------------------------------------------------------- --
     $${{o\operatorname{}\left( \frac{N^{2}\operatorname{}d^{2}}{N\operatorname{}d} \right)} = {o\operatorname{}{({N\operatorname{}d})}}}.$$   
  -- ----------------------------------------------------------------------------------------------------------------------------------------- --

However, the input to attention (matrices $\mathbf{Q},\mathbf{K},\mathbf{V}$) and the output $\mathbf{O}$ have size $N\operatorname{}d$ and they start out being
in HBM, so if the algorithm computes exact attention it must incur at least $\Omega\operatorname{}{({N\operatorname{}d})}$ HBM accesses. This is a
contradiction. ∎

###### Proof of [Theorem 5](#Thmtheorem5 "Theorem 5. ‣ B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

The IO complexity of the attention backward is very similar to the IO complexity of the attention forward
([Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).
Here we provide a sketch of the proof.

We first analyze the IO complexity of standard attention backward pass. The inputs
${\mathbf{Q},\mathbf{K},\mathbf{V},{\mathbf{d}\mathbf{O}}} \in {\mathbb{R}}^{N \times d}$ reside in HBM, and the at the end of the algorithm the outputs
${{\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}} \in {\mathbb{R}}^{N \times d}$ are written to HBM.

At each step of the standard attention backward pass, one needs to load inputs of size $N\operatorname{}d$ or $N^{2}$ from HBM, and needs to write the outputs
of size $N^{2}$ or $N\operatorname{}d$ to HBM. This incurs $\Theta\operatorname{}{({{N\operatorname{}d} + N^{2}})}$ HBM accesses.

We now analyze the IO complexity of FlashAttention backward pass.

Similar
to [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
we see that each element of $\mathbf{K}$ and $\mathbf{V}$ is loaded from HBM once. Each element of $\mathbf{d}\mathbf{K}$ and $\mathbf{d}\mathbf{V}$ is only
written to HBM once. We make $T_{c}$ passes over $\mathbf{Q},\mathbf{O},{\mathbf{d}\mathbf{O}}$, each pass loading all of
$\mathbf{Q},\mathbf{O},{\mathbf{d}\mathbf{O}}$ to HBM. We also make $T_{c}$ passes over $\mathbf{d}\mathbf{Q}$, each pass reading/writing all of
$\mathbf{d}\mathbf{Q}$ from/to HBM. Therefore the number of HBM accesses is
${\Theta\operatorname{}\left( {{N\operatorname{}d} + {N\operatorname{}d\operatorname{}T_{c}}} \right)} = {\Theta\operatorname{}{({N\operatorname{}d\operatorname{}T_{c}})}}$.

As in the proof
of [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
the constraints on the block sizes are that:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{B_{c} = {\Theta\operatorname{}\left( \frac{M}{d} \right)}},{B_{r} = {\Theta\operatorname{}\left( {\min\left( \frac{M}{d},d \right)} \right)}}}.$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------- --

We then have:

  -- ---------------------------------------------------------------------------------------------------- --
     $${T_{c} = \frac{N}{B_{c}} = {\Theta\operatorname{}\left( \frac{N\operatorname{}d}{M} \right)}}.$$   
  -- ---------------------------------------------------------------------------------------------------- --

As a result, the number of HBM accesses is:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------ --
     $${{\Theta\operatorname{}\left( {N\operatorname{}d\operatorname{}T_{c}} \right)} = {\Theta\operatorname{}\left( \frac{N^{2}\operatorname{}d^{2}}{M} \right)}}.$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------ --

∎

## Appendix D Extension Details

### D.1 Block-sparse FlashAttention

We describe the full block-sparse FlashAttention algorithm
in [Algorithm 5](#alg5 "Algorithm 5 ‣ D.1 Block-sparse FlashAttention ‣ Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
The algorithm is identical
to [Algorithm 2](#alg2 "Algorithm 2 ‣ B.3 FlashAttention: Forward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
except that we skip zero blocks.

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ in HBM, on-chip SRAM of size $M$, softmax scaling constant
$\tau \in {\mathbb{R}}$, masking function mask, dropout probability $p_{drop}$, block sizes
${B_{c} = \left\lceil \frac{M}{4\operatorname{}d} \right\rceil},{B_{r} = {\min\left( \left\lceil \frac{M}{4\operatorname{}d} \right\rceil,d \right)}}$, block
sparsity mask $M \in {\{ 0,1\}}^{{{N/B_{r}} \times N}/B_{c}}$..

1:  Initialize the pseudo-random number generator state $\mathcal{R}$ and save to HBM.

2:  Initialize
${\mathbf{O} = {(0)}_{N \times d} \in {\mathbb{R}}^{N \times d}},{\ell = {(0)}_{N} \in {\mathbb{R}}^{N}},{m = {({- \infty})}_{N} \in {\mathbb{R}}^{N}}$ in HBM.

3:  Divide $\mathbf{Q}$ into $T_{r} = \left\lceil \frac{N}{B_{r}} \right\rceil$ blocks $\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}$ of size $B_{r} \times d$ each,
and divide $\mathbf{K},\mathbf{V}$ in to $T_{c} = \left\lceil \frac{N}{B_{c}} \right\rceil$ blocks $\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}$ and
$\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}$, of size $B_{c} \times d$ each.

4:  Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}$ of size $B_{r} \times d$ each, divide $\ell$ into $T_{r}$ blocks
$\ell_{i},\ldots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\ldots,m_{T_{r}}$ of size $B_{r}$ each.

5:  for $1 \leq j \leq T_{c}$ do

6:     Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.

7:     for $1 \leq i \leq T_{r}$ do

8:        if $M_{i\operatorname{}j} \neq 0$ then

9:           Load $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.

10:           On chip, compute
$\mathbf{S}_{i\operatorname{}j} = {\tau\operatorname{}\mathbf{Q}_{i}\operatorname{}\mathbf{K}_{j}^{T}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

11:           On chip, compute $\mathbf{S}_{i\operatorname{}j}^{masked} = {\text{mask}\operatorname{}{(\mathbf{S}_{i\operatorname{}j})}}$.

12:           On chip, compute
${\overset{\sim}{m}}_{i\operatorname{}j} = {{rowmax}\operatorname{}{(\mathbf{S}_{i\operatorname{}j}^{masked})}} \in {\mathbb{R}}^{B_{r}}$,
${\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j} = {\exp{({\mathbf{S}_{i\operatorname{}j}^{masked} - {\overset{\sim}{m}}_{i\operatorname{}j}})}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$
(pointwise),
${\overset{\sim}{\ell}}_{i\operatorname{}j} = {{rowsum}\operatorname{}{({\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j})}} \in {\mathbb{R}}^{B_{r}}$.

13:           On chip, compute $m_{i}^{new} = {\max{(m_{i},{\overset{\sim}{m}}_{i\operatorname{}j})}} \in {\mathbb{R}}^{B_{r}}$,
$\ell_{i}^{new} = {{e^{m_{i} - m_{i}^{new}}\operatorname{}\ell_{i}} + {e^{{\overset{\sim}{m}}_{i\operatorname{}j} - m_{i}^{new}}\operatorname{}{\overset{\sim}{\ell}}_{i\operatorname{}j}}} \in {\mathbb{R}}^{B_{r}}$.

14:           On chip, compute
${\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j}^{dropped} = {{dropout}\operatorname{}{({\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j},p_{drop})}}$.

15:           Write
$\mathbf{O}_{i}\leftarrow{{diag}\operatorname{}{(\ell_{i}^{new})}^{- 1}\operatorname{}{({{{diag}\operatorname{}{(\ell_{i})}\operatorname{}e^{m_{i} - m_{i}^{new}}\operatorname{}\mathbf{O}_{i}} + {e^{{\overset{\sim}{m}}_{i\operatorname{}j} - m_{i}^{new}}\operatorname{}{\overset{\sim}{\mathbf{P}}}_{i\operatorname{}j}^{dropped}\operatorname{}\mathbf{V}_{j}}})}}$
to HBM.

16:           Write $\ell_{i}\leftarrow\ell_{i}^{new}$, $m_{i}\leftarrow m_{i}^{new}$ to HBM.

17:        end if

18:     end for

19:  end for

20:  Return $\mathbf{O},\ell,m,\mathcal{R}$.

Algorithm 5 Block-Sparse FlashAttention Forward Pass

We prove the IO-complexity of block-sparse FlashAttention.

###### Proof of [Proposition 4](#Thmtheorem4 "Proposition 4. ‣ 3.3 Extension: Block-Sparse FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.

The proof is very similar to the proof
of [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
For the block-sparse case, notice that we only need to load blocks corresponding to nonzero blocks. As a result, the number of HBM accesses are scaled by $s$,
the fraction of nonzero blocks in the block-sparsity mask. However, for small values of $s$, we would still need to write the result
$\mathbf{O} \in {\mathbb{R}}^{N \times d}$. Therefore the number of HBM accesses is

  -- -------------------------------------------------------------------------------------------------------------------------- --
     $${\Theta\operatorname{}\left( {{N\operatorname{}d} + {\frac{N^{2}\operatorname{}d^{2}}{M}\operatorname{}s}} \right)}.$$   
  -- -------------------------------------------------------------------------------------------------------------------------- --

∎

### D.2 Potential Extensions

We discuss here a few potential extensions of the IO-aware approach to speed up deep learning training.

Multi-GPU Attention. Large language models are trained on hundreds or thousands of GPUs, and one typically splits the attention computation between 4-8 GPUs on
the same node \[[77](#bib.bib77){.ltx_ref}\]. This introduces another level of memory hierarchy: beside GPU SRAM and GPU HBM, we also have the HBM of other
GPUs. For very long sequences, the different GPUs on the same node can cooperate to compute attention by taking into account the asymmetry of different levels
of memory hierarchy.

Sparse MLP layers. Typical dense MLP layers are compute-bound and not memory-bound. To improve their efficiency, MLP layers with sparse weight matrices can be
used \[[17](#bib.bib17){.ltx_ref}\]. However, many sparse MLP layers are instead memory-bound, and their speedup is often not proportional to the sparsity. We
believe that an IO-aware implementation can alleviate this issue and realize the benefits of sparsity. We are excited about future work in this direction, to
reduce the computational requirement of large models and improve their wall-block runtime.

Kernel machine learning. Our approach in FlashAttention relies on the fact that the $N \times N$ attention matrix is a function of a low-rank matrix
${\mathbf{Q}\mathbf{K}}^{\top}$ (of rank $d \ll N$). As a result, we can repeatedly load the inputs $\mathbf{Q},\mathbf{K}$ and recompute the block of the
attention matrix that we need, significantly reducing HBM access. As similar scenario happens in kernel machine learning: each element $K_{i\operatorname{}j}$
of the $N \times N$ kernel matrix $\mathbf{K}$ is a function of two vectors of size $d \ll N$, as it measures the similarity between two datapoints $x_{i}$ and
$x_{j}$. The KeOps library \[[26](#bib.bib26){.ltx_ref}, [8](#bib.bib8){.ltx_ref}\] is a successful example of how reducing memory reads/writes can speed up
kernel operations. We hope that this will motivate kernel methods that focus more on reducing IOs instead of just FLOPs.

## Appendix E Full Experimental Results

### E.1 BERT

We train BERT-large following the training procedure and hyperparameters of the reference MLPerf 1.1 implementation. In particular, we use the LAMB optimizer
with learning rate 3.75e-3, with batch size 448, trained for at most 7100 steps. The training is stopped once the validation accuracy (for masked language
modeling) reaches the target 72.0%, and the wall-clock run-time is measured. We train with FP16 precision using Apex AMP (with O2 optimization level).

We compare our results with the reported training speed from Nvidia that was submitted to MLPerf 1.1
([Table 1](#S4.T1 "Table 1 ‣ BERT. ‣ 4.1 Faster Models with FlashAttention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).

We use the same train / validation data split provided by MLPerf 1.1 reference implementation. In particular, we evaluate on the same 10000 validation examples
as the baseline from Nvidia.

We train the model on 8$\times$A100-80GB GPUs. Each training run takes between 16 and 19 minutes, and we average the results of 10 runs.

### E.2 GPT-2

We use the standard implementations of GPT-2 \[[67](#bib.bib67){.ltx_ref}\] from Huggingface transformers library and from Nvidia's Megatron-LM repo. We follow
the training recipe of the Megatron-LM repo.

We use an effective batch size of 512, and use gradient accumulation to fit into available GPU memory. We use the AdamW optimizer, with learning rate 6e-4 for
GPT-2 small and 1.5e-4 for GPT-2 medium, and weight decay of 0.1. All models are trained with the same hyperparameters for 400K steps. We run all
implementations with mixed-precision training (PyTorch AMP).

We use the Openwebtext dataset, with the GPT-2 BPE tokenizer. We randomly select 0.5% of the dataset as the validation set, with the rest being used as training
set. This random selection of validation set is done once, and all models are evaluated on the same validation set.

We train the model on 8$\times$A100-40GB GPUs, and we measure the wall-clock training time. Training GPT-2 small takes between 2.7-9.5 days, and training GPT-2
medium takes between 6.9-21.0 days
([Table 2](#S4.T2 "Table 2 ‣ GPT-2. ‣ 4.1 Faster Models with FlashAttention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}).

In [Fig. 4](#A5.F4 "Figure 4 ‣ E.2 GPT-2 ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
we plot of the validation perplexity throughout training of GPT-2 small/medium, using either HuggingFace implementation or our FlashAttention implementation. We
see that FlashAttention behaves the same as the baseline implementation and the validation perplexity curves of the two implementations almost lie on top of
each other.

![Refer to caption](/html/2205.14135/assets/x4.png){#A5.F4.g1 .ltx_graphics .ltx_centering .ltx_img_landscape width="629" height="490"}

Figure 4: Validation perplexity of GPT-2 small/medium using two implementations. We confirm that FlashAttention yields the same validation curves as the
baseline implementation from HuggingFace.

##### Long Document Classification.

For MIMIC-III and ECtHR, we follow the hyperparameters of Dai et al. \[[13](#bib.bib13){.ltx_ref}\].

### E.3 LRA details

We follow the hyperparameters from the Long-range arena paper \[[80](#bib.bib80){.ltx_ref}\], the Long-range arena repo
([https://github.com/google-research/long-range-arena](https://github.com/google-research/long-range-arena){.ltx_ref .ltx_url .ltx_font_typewriter
target="_blank"}), and the Nyströmformer reproduction \[[90](#bib.bib90){.ltx_ref}\]. To be generous to the baseline methods, if we are unable to reproduce the
performance of any baseline for any of the five tasks, we report the better performance from Tay et al. \[[80](#bib.bib80){.ltx_ref}\] or Xiong et al.
\[[90](#bib.bib90){.ltx_ref}\] for that baseline on that task.

After hyperparameter tuning, almost all of the attention methods achieve similar accuracy on all of the five LRA tasks.

We run all methods with mixed-precision training, except for Performer (not stable with mixed precision) and Local Attention (implementation does not support
FP16).

To calculate the overall wallclock-time speedup, we take the geometric mean of the wallclock-time speedup of each of the five tasks.

##### Path-X

For Path-X and Path-256, we follow the hyperparameters from the PathFinder-32 experiments from the long-range arena paper\[[80](#bib.bib80){.ltx_ref}\]. For
both, we first pretrain a model on Path-64. We take the checkpoint after 200 epochs, upsample its positional embedding (we duplicate the positional embeddings
gridwise in space), and fine-tune it on the downstream task for 200 epochs with one epoch of linear warmup, and cosine decay of the learning rate. For Path-X,
we take the best performing checkpoint (according to val accuracy), and additionally fine-tune it for 200 epochs with the same warmup and learning rate (this
adds roughly 4 points of accuracy to FlashAttention for Path-X, but the model starts overfitting afterwards).

### E.4 Comparison with Apex FMHA

We compare our method/implementation with Apex FMHA
([https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha){.ltx_ref .ltx_url
.ltx_font_typewriter target="_blank"}).

When we started this project, Apex FMHA was the fastest implementation of attention (that we knew of), tailored for short sequences of length at most 512. In
fact, almost all MLPerf submissions for BERT training benchmark running on Nvidia GPUs use FMHA for their model code, as of MLPerf
1.1 \[[58](#bib.bib58){.ltx_ref}\]. Since FMHA targets BERT models, it only supports head dimension 64, and only runs on A100 GPUs. FMHA fuses the attention
computation ${dropout}\operatorname{}{({{softmax}\operatorname{}{({\text{mask}\operatorname{}{({\mathbf{Q}\mathbf{K}}^{\top})}})}})}\operatorname{}\mathbf{V}$
into one CUDA kernel. In the forward pass, it stores the attention matrix
${softmax}\operatorname{}{({\text{mask}\operatorname{}{({\mathbf{Q}\mathbf{K}}^{T})}})}$ to HBM to be used in gradient computation. As a result, it does not
offer substantial memory saving (though for shorter sequences memory footprint is often not a primary concern).

We use FMHA code as a starting point, and apply two well-established techniques (tiling and recomputation) to deal with long sequences and to save memory as
mentioned
in [Section 3](#S3 "3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
As a result, we can support much longer sequences (e.g., up to length 64K). We also support more head dimensions (16, 32, 64, 128) and broader GPU types (all
Turing and Ampere GPUs at the time of writing).

In [Table 7](#A5.T7 "Table 7 ‣ E.4 Comparison with Apex FMHA ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref},
we compare the performance of FlashAttention and Apex FMHA for short sequences (as FMHA only supports sequence length at most 512). Generally FlashAttention is
slightly faster than FMHA in the forward pass and slightly slower than FMHA in the backward pass. This is because we do not store the attention matrix in the
forward pass and recompute it in the backward pass. Compared to FMHA, the overall runtime of FlashAttention is about 4% slower for sequence length 128, 8%
faster for sequence length 256, and 5% faster for sequence length 512.

  Attention Method                    128    256    512
  ----------------------------------- ------ ------ ------
  Apex FMHA forward                   0.10   0.29   1.14
  FlashAttention forward              0.08   0.22   0.81
  Apex FMHA backward                  0.17   0.52   1.81
  FlashAttention backward             0.20   0.53   2.00
  Apex FMHA forward + backward        0.27   0.81   2.95
  FlashAttention forward + backward   0.28   0.75   2.81

Table 7: Runtime (ms) of FlashAttention compared to FMHA by sequence length, with masking and dropout, measured on an A100-SXM4-40GB GPU. Batch size 64, 16
heads, head dimension 64 (i.e., BERT-large size).

### E.5 Speedup On Different Hardware and Configurations

Speedup varies between different types of GPU types and generations depending on HBM bandwidth and SRAM size. In this section, we profile FlashAttention speedup
on different GPUs and configurations.

![Refer to caption](/html/2205.14135/assets/figs/flashattn_speedup.jpg){#A5.F5.g1 .ltx_graphics .ltx_centering .ltx_img_landscape width="548" height="254"}

Figure 5: Speedup over standard PyTorch attention at different sequence lengths, on A100.

##### A100

Figure [5](#A5.F5 "Figure 5 ‣ E.5 Speedup On Different Hardware and Configurations ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows speedup on an A100 GPU with batch size 8, head dimension 64, and 12 attention heads, across different sequence lengths. We generally see 2-4$\times$
speedup, and we see more speedup when using dropout and masking due to kernel fusion.

![Refer to caption](/html/2205.14135/assets/figs/flashattn_speedup_a100_d128.jpg){#A5.F6.g1 .ltx_graphics .ltx_centering .ltx_img_landscape width="548"
height="241"}

Figure 6: Speedup over standard PyTorch attention at different sequence lengths, on A100, with head dimension 128.

##### A100, Head Dimension 128

Speedup also changes when we increase the head dimension. Each block requires more memory, so we need to use smaller block sizes to fit into SRAM.
Figure [6](#A5.F6 "Figure 6 ‣ A100 ‣ E.5 Speedup On Different Hardware and Configurations ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows speedup with head dimension 128 on an A100 (batch size 16, 12 heads). We see less speedup overall---but we can still see significant speedup (up to
3$\times$) with a causal mask, where half the blocks are masked out.

![Refer to caption](/html/2205.14135/assets/figs/flashattn_speedup_3090.jpg){#A5.F7.g1 .ltx_graphics .ltx_centering .ltx_img_landscape width="548" height="254"}

Figure 7: Speedup over standard PyTorch attention at different sequence lengths, on RTX 3090.

##### RTX 3090

Figure [7](#A5.F7 "Figure 7 ‣ A100, Head Dimension 128 ‣ E.5 Speedup On Different Hardware and Configurations ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows speedup on an RTX 3090 GPU. Here, we use batch size 12 with 12 attention heads. We observe slightly higher speedups on the RTX 3090 (between
2.5-4.5$\times$), since the memory bandwidth on an RTX 3090 is lower than on an A100 (roughly 900 GB/s vs. 1.5 TB/s).

![Refer to caption](/html/2205.14135/assets/figs/flashattn_speedup_t4.jpg){#A5.F8.g1 .ltx_graphics .ltx_centering .ltx_flex_size_1 .ltx_img_landscape
width="548" height="249"}

![Refer to caption](/html/2205.14135/assets/figs/flashattn_speedup_t4_fwd.jpg){#A5.F8.g2 .ltx_graphics .ltx_centering .ltx_flex_size_1 .ltx_img_landscape
width="548" height="254"}

Figure 8: Speedup over standard PyTorch attention at different sequence lengths, on T4. Top: Combined forward pass + backward pass. Bottom: Forward pass only.

##### T4

Figure [8](#A5.F8 "Figure 8 ‣ RTX 3090 ‣ E.5 Speedup On Different Hardware and Configurations ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
shows speedup on a T4 GPU. T4 SRAM is smaller than A100, so we need to make the block sizes smaller in FlashAttention. As a result, we observe less speedup on
T4, which matches the IO complexity analysis in
Section [3.2](#S3.SS2 "3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}.
T4 GPUs are commonly used for inference, so we also report speedup on the forward pass only.

### E.6 Full Benchmarking Results

We report the full benchmarking results and experimental details on A100.

##### Baselines

We compare against reference implementations for exact attention from PyTorch/HuggingFace and Megatron, approximate attention, and sparse attention. For
approximate attention, we compare against reference implementations of Reformer \[[51](#bib.bib51){.ltx_ref}\], Local Attention \[[68](#bib.bib68){.ltx_ref}\],
Linformer Attention \[[84](#bib.bib84){.ltx_ref}\], Smyrf \[[19](#bib.bib19){.ltx_ref}\], and LongShortFormer (LSFormer) \[[94](#bib.bib94){.ltx_ref}\]. For
sparse attention, we compare against reference implementations of Block-Sparse Attention form OpenAI \[[11](#bib.bib11){.ltx_ref}\],
Longformer\[[3](#bib.bib3){.ltx_ref}\], and BigBird Attention \[[92](#bib.bib92){.ltx_ref}\]. For the approximate and sparse attention, we use a compression
ratio of 1/8, or a compressed sequence length of 256, whichever is smaller.

##### Setup

We measure runtime and memory usage of the attention computation with 8 heads of dimension 64, and batch size 16 on a machine with one A100 GPU with 40 GB of
GPU HBM. We vary sequence length in our experiments. We compute attention on random vectors for $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ (we do not measure
the projection from the hidden layer). For dropout, we use dropout 0.1; for masking, we use a padding mask with uniformly-random mask lengths between the total
sequence length and the total sequence length minus 20. To measure runtime, we take the average of 100 measurements of the attention call. We only measure
memory footprint once, since it does not vary between runs.

We report timing results on the forward pass, backward pass, and combined forward + backward pass. We measure each method with and without dropout, masking, or
both---except for Block Sparse, Longformer, and BigBird. These methods did not successfully run the backward pass with masking due to a bug in external
libraries, so we measured them without masking to be generous. We use FP16 for all measurements, except for Local Attention, whose implementation only supports
FP32.

For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not
support sequence lengths longer than 2048. Block-Sparse (OpenAI) does not support sequence lengths longer than 4096. Longformer and BigBird do not support
sequence lengths longer than 8092.

We measure memory usage on the combined forward + backward pass, without dropout or masking.

##### Results

[Table 8](#A5.T8 "Table 8 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
summarizes all the experimental configurations and contains pointers to the results tables.

  Dropout   Masking   Pass                      Table
  --------- --------- ------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Yes       Yes       Forward                   [Table 9](#A5.T9 "Table 9 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  Yes       Yes       Backward                  [Table 10](#A5.T10 "Table 10 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  Yes       Yes       Combined                  [Table 11](#A5.T11 "Table 11 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  No        Yes       Forward                   [Table 12](#A5.T12 "Table 12 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  No        Yes       Backward                  [Table 13](#A5.T13 "Table 13 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  No        Yes       Combined                  [Table 14](#A5.T14 "Table 14 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  Yes       No        Forward                   [Table 15](#A5.T15 "Table 15 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  Yes       No        Backward                  [Table 16](#A5.T16 "Table 16 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  Yes       No        Combined                  [Table 17](#A5.T17 "Table 17 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  No        No        Forward                   [Table 18](#A5.T18 "Table 18 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  No        No        Backward                  [Table 19](#A5.T19 "Table 19 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  No        No        Combined                  [Table 20](#A5.T20 "Table 20 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}
  No        No        Memory Usage (Combined)   [Table 21](#A5.T21 "Table 21 ‣ Results ‣ E.6 Full Benchmarking Results ‣ Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"){.ltx_ref}

Table 8: Pointers to results tables.

  ----------------------------- ------ ------ ------ ------- ------- ------- ------- -------- -------- ---------
  Attention Method              128    256    512    1024    2048    4096    8192    16384    32768    65536
  PyTorch Attention             0.36   0.34   0.78   2.54    9.33    36.33   -       -        -        -
  Megatron                      0.40   0.40   1.10   3.65    16.19   -       -       -        -        -
  Reformer                      2.03   3.15   5.67   11.02   22.59   46.14   97.38   212.13   -        -
  Local Attention               0.83   0.86   1.01   2.20    7.13    14.32   28.60   57.79    117.67   -
  Linformer                     0.67   0.52   0.69   0.71    1.65    3.18    6.15    12.16    24.17    52.39
  Smyrf                         2.27   2.34   3.91   7.44    14.71   29.22   58.27   116.41   -        -
  LSformer                      1.18   1.27   1.34   3.38    11.40   22.55   44.95   89.76    179.66   -
  Block Sparse                  1.12   1.11   2.13   2.77    6.95    20.91   -       -        -        -
  Longformer                    1.22   1.14   1.08   1.95    5.72    12.98   -       -        -        -
  BigBird                       1.13   1.12   1.12   1.77    6.03    13.68   -       -        -        -
  FlashAttention                0.04   0.06   0.21   0.82    2.85    10.41   41.74   167.19   670.76   2682.35
  Block-Sparse FlashAttention   0.06   0.06   0.06   0.12    0.44    0.86    1.70    3.29     6.55     13.34
  ----------------------------- ------ ------ ------ ------- ------- ------- ------- -------- -------- ---------

Table 9: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second
best underlined.

  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------
  Attention Method              128    256    512    1024    2048    4096    8192     16384    32768     65536
  PyTorch Attention             0.37   0.49   1.66   5.81    22.32   87.67   -        -        -         -
  Megatron                      0.35   0.32   0.77   2.42    8.43    -       -        -        -         -
  Reformer                      2.37   4.59   8.91   17.68   35.13   70.05   140.01   -        -         -
  Local Attention               0.55   0.62   1.49   4.03    13.78   27.61   55.20    110.27   221.40    -
  Linformer                     0.89   0.80   0.81   0.93    2.48    4.75    9.29     18.27    36.53     -
  Smyrf                         1.41   2.83   5.43   10.72   21.25   42.31   84.48    168.95   -         -
  LSformer                      1.75   1.76   3.01   7.50    20.07   39.08   76.39    150.82   -         -
  Block Sparse                  1.29   1.28   2.18   3.04    7.27    21.16   -        -        -         -
  Longformer                    1.27   1.31   1.29   2.04    5.24    10.74   25.95    -        -         -
  BigBird                       1.33   1.28   1.32   1.81    5.55    11.44   27.45    -        -         -
  FlashAttention                0.30   0.26   0.68   2.02    6.84    26.89   105.70   418.96   1666.89   6660.44
  Block-Sparse FlashAttention   0.30   0.27   0.29   0.59    1.50    2.94    5.82     11.85    23.98     47.61
  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------

Table 10: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second
best underlined.

  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------
  Attention Method              128    256    512     1024    2048    4096     8192     16384    32768     65536
  PyTorch Attention             0.84   0.86   2.35    8.29    31.75   124.19   -        -        -         -
  Megatron                      0.87   0.89   1.33    4.21    16.50   -        -        -        -         -
  Reformer                      4.30   7.76   14.60   28.74   57.79   116.34   237.57   -        -         -
  Local Attention               1.40   1.60   2.06    6.06    20.94   42.01    84.08    168.48   339.45    -
  Linformer                     1.57   1.49   1.55    1.60    4.19    8.04     15.71    30.92    61.47     -
  Smyrf                         3.41   5.08   9.35    18.18   36.03   71.68    143.04   285.87   -         -
  LSformer                      3.08   3.10   4.26    10.90   31.59   61.72    121.51   241.18   -         -
  Block Sparse                  2.54   2.52   3.71    5.44    13.29   39.19    -        -        -         -
  Longformer                    2.47   2.49   2.51    3.10    10.39   22.49    60.44    -        -         -
  BigBird                       2.51   2.49   2.52    3.40    10.97   23.89    63.28    -        -         -
  FlashAttention                0.43   0.41   0.95    2.55    9.56    37.49    147.75   586.61   2339.11   9341.30
  Block-Sparse FlashAttention   0.44   0.44   0.45    0.89    1.95    4.12     7.64     16.60    32.73     64.11
  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------

Table 11: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best
in bold, second best underlined.

  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- -------- ---------
  Attention Method              128    256    512    1024    2048    4096    8192     16384    32768    65536
  PyTorch Attention             0.30   0.30   0.63   1.93    7.08    27.45   112.90   -        -        -
  Megatron                      0.45   0.41   0.43   1.52    5.80    -       -        -        -        -
  Reformer                      1.87   3.00   5.37   10.43   21.40   43.83   92.80    203.24   -        -
  Local Attention               0.70   0.81   1.02   2.09    6.64    13.34   26.77    54.02    110.11   -
  Linformer                     0.63   0.50   0.67   0.65    1.36    2.60    5.04     9.92     19.69    43.47
  Smyrf                         2.38   2.32   3.76   7.16    14.14   28.09   55.98    111.73   -        -
  LSformer                      1.22   1.29   1.44   3.28    10.99   21.72   43.29    86.32    172.76   -
  Block Sparse                  0.96   1.04   1.66   2.16    5.41    16.15   -        -        -        -
  Longformer                    0.99   0.98   0.99   1.56    4.79    11.07   32.98    -        -        -
  BigBird                       0.96   1.02   1.02   1.48    5.05    11.59   34.16    -        -        -
  FlashAttention                0.03   0.04   0.17   0.68    2.28    8.40    33.55    134.14   537.50   2150.88
  Block-Sparse FlashAttention   0.05   0.04   0.05   0.11    0.35    0.68    1.33     2.54     5.34     10.73
  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- -------- ---------

Table 12: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best
underlined.

  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------
  Attention Method              128    256    512    1024    2048    4096    8192     16384    32768     65536
  PyTorch Attention             0.44   0.46   1.53   5.33    20.34   79.87   -        -        -         -
  Megatron                      0.29   0.31   0.65   1.95    6.49    -       -        -        -         -
  Reformer                      2.31   4.47   8.68   17.20   34.14   68.09   136.02   -        -         -
  Local Attention               0.51   0.62   1.30   3.81    13.33   26.72   53.41    106.82   214.15    -
  Linformer                     0.76   0.81   0.94   0.87    2.24    4.25    8.35     16.38    32.67     72.11
  Smyrf                         1.34   2.77   5.30   10.46   20.73   41.27   82.41    164.86   -         -
  LSformer                      1.66   1.61   3.09   7.42    19.68   38.35   74.92    147.86   -         -
  Block Sparse                  1.24   1.25   2.04   2.91    6.78    19.67   -        -        -         -
  Longformer                    1.27   1.23   1.24   1.85    4.99    10.21   24.89    -        -         -
  BigBird                       1.43   1.50   1.44   1.69    5.25    10.86   26.26    -        -         -
  FlashAttention                0.21   0.22   0.62   1.84    5.77    22.25   86.21    338.91   1343.91   5361.09
  Block-Sparse FlashAttention   0.22   0.22   0.26   0.57    1.55    3.13    5.98     12.21    23.49     47.85
  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------

Table 13: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best
underlined.

  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------
  Attention Method              128    256    512     1024    2048    4096     8192     16384    32768     65536
  PyTorch Attention             0.80   0.81   2.08    7.23    27.51   107.58   -        -        -         -
  Megatron                      0.81   0.83   1.09    3.36    12.39   -        -        -        -         -
  Reformer                      4.16   7.46   14.06   27.68   55.66   112.15   229.37   -        -         -
  Local Attention               1.39   1.68   2.08    5.83    20.04   40.16    80.44    161.35   325.11    -
  Linformer                     1.51   1.42   1.56    1.67    3.67    6.99     13.63    26.77    53.36     117.56
  Smyrf                         3.38   4.93   9.07    17.66   34.94   69.55    138.72   277.41   -         -
  LSformer                      3.08   3.10   4.26    10.90   31.59   61.72    121.51   241.18   -         -
  Block Sparse                  2.39   2.40   3.31    5.02    12.25   35.94    -        -        -         -
  Longformer                    2.36   2.34   2.38    2.94    9.83    21.35    58.12    -        -         -
  BigBird                       2.35   2.35   2.37    3.25    10.36   22.57    60.63    -        -         -
  FlashAttention                0.32   0.30   0.83    2.37    7.95    30.77    119.98   473.65   1883.43   7513.01
  Block-Sparse FlashAttention   0.34   0.34   0.36    0.69    1.85    3.89     7.16     14.85    30.46     60.03
  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------

Table 14: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold,
second best underlined.

  ----------------------------- ------ ------ ------ ------- ------- ------- ------- -------- -------- ---------
  Attention Method              128    256    512    1024    2048    4096    8192    16384    32768    65536
  PyTorch Attention             0.26   0.24   0.57   1.80    6.56    25.34   -       -        -        -
  Megatron                      0.27   0.27   0.56   1.88    6.56    -       -       -        -        -
  Reformer                      1.83   2.96   5.31   10.33   21.19   43.42   91.96   201.34   -        -
  Local Attention               0.51   0.60   0.78   2.01    6.23    12.52   25.07   50.50    102.18   -
  Linformer                     0.47   0.37   0.49   0.52    1.37    2.65    5.12    10.13    20.25    44.16
  Smyrf                         2.12   2.01   3.15   5.97    11.83   23.36   46.48   92.72    -        -
  LSformer                      1.28   1.33   1.51   3.39    11.40   22.54   44.96   89.85    179.73   -
  Block Sparse                  1.03   1.00   1.72   2.39    5.96    17.88   -       -        -        -
  Longformer                    1.02   1.03   1.03   1.73    5.10    11.63   34.22   -        -        -
  BigBird                       0.99   1.03   1.01   1.58    5.36    12.27   35.56   -        -        -
  FlashAttention                0.10   0.10   0.22   0.83    2.81    10.38   41.63   167.01   668.74   2678.11
  Block-Sparse FlashAttention   0.54   0.51   0.68   0.61    0.67    1.10    1.89    3.71     7.18     14.41
  ----------------------------- ------ ------ ------ ------- ------- ------- ------- -------- -------- ---------

Table 15: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best
underlined.

  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------
  Attention Method              128    256    512    1024    2048    4096    8192     16384    32768     65536
  PyTorch Attention             0.44   0.35   0.90   2.94    10.77   41.67   -        -        -         -
  Megatron                      0.28   0.33   0.92   2.94    10.80   -       -        -        -         -
  Reformer                      2.24   4.34   8.39   16.62   33.02   65.77   131.52   -        -         -
  Local Attention               0.51   0.58   1.41   3.71    12.96   25.98   51.94    103.72   207.78    -
  Linformer                     0.84   0.74   0.79   0.85    2.28    4.37    8.66     17.02    33.78     -
  Smyrf                         1.27   2.56   4.90   9.66    19.16   38.13   76.17    152.39   -         -
  LSformer                      1.67   1.77   3.03   7.52    20.10   39.13   76.35    150.83   -         -
  Block Sparse                  1.27   1.36   2.15   3.04    7.27    21.18   -        -        -         -
  Longformer                    1.28   1.34   1.38   1.98    5.24    10.74   25.95    -        -         -
  BigBird                       1.48   1.47   1.50   1.81    5.57    11.38   27.43    -        -         -
  FlashAttention                0.15   0.18   0.58   1.86    6.50    26.21   104.27   416.10   1661.92   6643.01
  Block-Sparse FlashAttention   0.17   0.17   0.17   0.40    1.10    2.04    4.43     9.33     18.28     37.31
  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------

Table 16: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best
underlined.

  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------
  Attention Method              128    256    512     1024    2048    4096     8192     16384    32768     65536
  PyTorch Attention             0.66   0.67   1.43    4.82    17.47   67.29    -        -        -         -
  Megatron                      0.88   0.90   1.49    4.73    17.41   -        -        -        -         -
  Reformer                      4.06   7.28   13.68   26.98   54.27   109.39   223.80   -        -         -
  Local Attention               1.09   1.40   1.99    5.61    19.23   38.62    77.30    154.63   311.12    -
  Linformer                     1.31   1.21   1.30    1.39    3.73    7.15     14.05    27.69    55.00     -
  Smyrf                         3.00   4.37   8.05    15.66   31.04   61.64    123.04   245.65   -         -
  LSformer                      3.07   3.17   4.31    10.89   31.54   61.78    121.56   240.94   -         -
  Block Sparse                  2.54   2.52   3.71    5.44    13.29   39.19    -        -        -         -
  Longformer                    2.47   2.49   2.51    3.10    10.39   22.49    60.44    -        -         -
  BigBird                       2.51   2.49   2.52    3.40    10.97   23.89    63.28    -        -         -
  FlashAttention                0.35   0.36   0.80    2.52    9.16    36.70    146.13   583.45   2332.01   9323.63
  Block-Sparse FlashAttention   0.91   0.83   0.94    0.92    1.83    3.50     7.02     13.56    26.71     53.92
  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------

Table 17: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold,
second best underlined.

  ----------------------------- ------ ------ ------ ------ ------- ------- ------- -------- -------- ---------
  Attention Method              128    256    512    1024   2048    4096    8192    16384    32768    65536
  PyTorch Attention             0.21   0.22   0.43   1.27   4.32    16.47   67.77   -        -        -
  Megatron                      0.24   0.26   0.42   1.33   4.28    -       -       -        -        -
  Reformer                      1.77   2.82   5.01   9.74   20.03   41.11   87.39   192.40   -        -
  Local Attention               0.48   0.57   0.80   1.90   5.76    11.56   23.13   46.65    94.74    -
  Linformer                     0.46   0.36   0.45   0.50   1.09    2.09    4.01    7.90     15.70    35.40
  Smyrf                         1.94   1.96   3.01   5.69   11.26   22.23   44.21   88.22    -        -
  LSformer                      1.21   1.34   1.34   3.31   11.01   21.71   43.27   86.32    172.85   -
  Block Sparse                  0.96   1.04   1.66   2.16   5.41    16.15   -       -        -        -
  Longformer                    0.99   0.98   0.99   1.56   4.79    11.07   32.98   -        -        -
  BigBird                       0.96   1.02   1.02   1.48   5.05    11.59   34.16   -        -        -
  FlashAttention                0.08   0.09   0.18   0.68   2.40    8.42    33.54   134.03   535.95   2147.05
  Block-Sparse FlashAttention   0.56   0.52   0.63   0.65   0.61    0.96    1.69    3.02     5.69     11.77
  ----------------------------- ------ ------ ------ ------ ------- ------- ------- -------- -------- ---------

Table 18: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------
  Attention Method              128    256    512    1024    2048    4096    8192     16384    32768     65536
  PyTorch Attention             0.26   0.29   0.78   2.44    8.82    33.87   -        -        -         -
  Megatron                      0.29   0.30   0.80   2.59    8.86    -       -        -        -         -
  Reformer                      2.18   4.21   8.14   16.12   32.02   63.84   127.60   -        -         -
  Local Attention               0.51   0.64   1.28   3.60    12.52   25.08   50.22    100.23   200.66    -
  Linformer                     0.69   0.76   0.69   0.80    2.04    3.88    7.67     15.04    30.11     63.15
  Smyrf                         1.24   2.49   4.77   9.42    18.65   37.12   74.15    148.35   -         -
  LSformer                      1.68   1.61   3.02   7.40    19.72   38.27   74.89    147.99   -         -
  Block Sparse                  1.24   1.25   2.04   2.91    6.78    19.67   -        -        -         -
  Longformer                    1.27   1.23   1.24   1.85    4.99    10.21   24.89    -        -         -
  BigBird                       1.43   1.50   1.44   1.69    5.25    10.86   26.26    -        -         -
  FlashAttention                0.11   0.16   0.52   1.62    5.45    21.57   84.75    336.00   1338.56   5343.19
  Block-Sparse FlashAttention   0.11   0.12   0.16   0.38    1.20    2.34    4.69     9.10     18.74     37.04
  ----------------------------- ------ ------ ------ ------- ------- ------- -------- -------- --------- ---------

Table 19: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------
  Attention Method              128    256    512     1024    2048    4096     8192     16384    32768     65536
  PyTorch Attention             0.67   0.70   1.18    3.67    13.22   50.44    -        -        -         -
  Megatron                      0.74   0.65   1.23    3.80    13.21   -        -        -        -         -
  Reformer                      3.93   7.01   13.15   25.89   52.09   105.00   215.13   -        -         -
  Local Attention               1.09   1.27   1.99    5.38    18.32   36.77    73.67    147.29   296.35    -
  Linformer                     1.31   1.25   1.30    1.29    3.20    6.10     11.93    23.39    46.72     100.52
  Smyrf                         2.98   4.23   7.78    15.12   29.96   59.45    118.60   237.02   -         -
  LSformer                      3.03   3.05   4.26    10.70   30.77   60.15    118.33   234.94   -         -
  Block Sparse                  2.39   2.40   3.31    5.02    12.25   35.94    -        -        -         -
  Longformer                    2.36   2.34   2.38    2.94    9.83    21.35    58.12    -        -         -
  BigBird                       2.35   2.35   2.37    3.25    10.36   22.57    60.63    -        -         -
  FlashAttention                0.31   0.31   0.73    2.29    7.64    30.09    118.50   470.51   1876.08   7492.85
  Block-Sparse FlashAttention   0.74   0.77   0.82    0.88    1.71    3.21     6.56     12.60    24.93     50.39
  ----------------------------- ------ ------ ------- ------- ------- -------- -------- -------- --------- ---------

Table 20: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best
underlined.

  ----------------------------- ----- ----- ------ ------ ------ ------- ------- ------- ------- -------
  Attention Method              128   256   512    1024   2048   4096    8192    16384   32768   65536
  PyTorch Attention             36    104   336    1184   4416   17024   -       -       -       -
  Megatron                      36    104   336    1184   4416   -       -       -       -       -
  Reformer                      377   754   1508   3016   6033   12067   24134   -       -       -
  Local Attention               53    110   232    592    1696   3392    6784    13568   27136   -
  Linformer                     25    52    114    287    832    1652    3292    6572    13132   26252
  Smyrf                         217   434   868    1737   3474   6947    13894   27788   -       -
  LSformer                      72    152   333    796    2540   5068    10125   20240   -       -
  Block Sparse                  33    82    228    408    910    2401    -       -       -       -
  Longformer                    30    61    124    277    681    1370    2748    -       -       -
  BigBird                       33    66    131    294    708    1431    2872    -       -       -
  FlashAttention                22    44    104    209    418    836     1672    3344    6688    13376
  Block-Sparse FlashAttention   22    44    104    209    418    836     1672    3344    6690    13384
  ----------------------------- ----- ----- ------ ------ ------ ------- ------- ------- ------- -------

Table 21: Memory usage (MB) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.