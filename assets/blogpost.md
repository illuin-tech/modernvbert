# ModernVBERT: Towards Smaller Visual Document Retrievers

![thumbnail](https://cdn-uploads.huggingface.co/production/uploads/6651baf4b34bbdaec88333e7/aPgDgO53qaI8kUA28L49D.png)

## TL;DR
This blog post introduces ModernVBERT, a 250M vision-language retriever, state-of-the-art for its size category on document retrieval. ModernVBERT is small, fast, and fully open-source. All model checkpoints, training datasets and training recipes are released under MIT license.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6651baf4b34bbdaec88333e7/r9J8hirrRe0NmpATKDMvz.png" alt="modernvbert pareto frontier" width="70%">
</p>


## What’s The Matter With Current Retrievers?
Most recent visual encoders are built by adapting large vision-language models (VLMs). These models dominate retrieval benchmarks, surpassing contrastively pretrained dual encoders by leveraging their strength in complex multimodal reasoning and their ability to transfer knowledge across diverse visual tasks.

Yet, they inherit a fundamental limitation from their generative roots: **causal attention**. Designed for text generation, VLMs process tokens only in a forward direction, predicting the next token from past ones. While effective for generation, this hampers embedding quality for retrieval. By ignoring future context, they might fail to capture relationships that retrieval requires.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6651baf4b34bbdaec88333e7/ghbNwoG9qO5tEcHTxIweG.png" alt="causal vs bidirectional attention">
</p>

That observation led us to a crucial question: *What if we built a visual retriever from scratch, with bidirectional attention at its core?*

## Bidirectional or Not Bidirectional?

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6651baf4b34bbdaec88333e7/5_dKoOOi9c7xWHVVMx0tQ.png" alt="causal masks vidore" width="70%">
</p>

So, does bidirectional attention really make a difference for retrieval? To find out, we set up a controlled experiment where the only change was the attention scheme across models. The results were surprising: in single-vector retrieval, the gap was negligible. But once we moved to multi-vector settings the benefits of bidirectional attention jumped out. On standard document retrieval benchmarks with Late Interaction matching, bidirectional encoders scored a **+10.6 nDCG@5 boost** over causal decoders. 

<blockquote>
<b>Takeaway:</b> Bidirectional attention is crucial to unlock the full power of Late Interaction.
</blockquote>

## Training Tricks That Make the Difference

Bidirectional attention is only half the story. In practice, other training choices can have major impact on downstream performance:

**High-Resolution Inputs for Document Tasks**

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6651baf4b34bbdaec88333e7/n61RjEICd9ATlJoLksuk8.jpeg" alt="image resolution impact on retrieval" width="80%">
</p>

When it comes to documents, detail matters. Training with higher image resolutions (up to 2048px) consistently improved performance on document retrieval tasks. Additional gains were observed when increasing the resolution at the end of the modality alignment phase (HR cooldown), yielding a +49.2% relative improvement compared to the visual encoder resolution. This allowed the model to better capture fine-grained text and layout signals.

**Smart Data Mixing Pays Off**

Not enough “document + query” pairs? No problem. By augmenting with text-only pairs, we gained **+1.7 nDCG@5** on document retrieval benchmarks. The model was able to transfer its understanding across modalities, and since text-only training is cheap and scalable, it boosted performance without slowing training.

<blockquote>
<b>Takeaway:</b> It isn’t just about architecture, it’s also about thoughtful training choices. Small tweaks, like resolution and data mixing, can compound into important gains.
</blockquote>

## Here Comes ModernVBERT
With all that in mind, we scale up the recipe to train ModernVBERT, a 250M parameter visual encoder based on the text-encoder Ettin-150M (ModernBERT architecture).

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6651baf4b34bbdaec88333e7/KIPlY5Bt1uFqgOuPgOTB1.jpeg" alt="image resolution impact on retrieval" width="80%">
</p>

The outcome? When finetuned for visual document retrieval tasks, ModernVBERT matches the performance of models nearly **10x larger** on visual document benchmarks. Additionally, it provides an interesting inference speed on CPU compared to the models of similar performance.

## Why Small Retrievers Matters?
While large VLM-based retrievers still achieve state-of-the-art results, they’re often impractical in real-world scenarios where expensive compute resources are not available at inference. Thanks to its compact design, ModernVBERT enables fast CPU query encoding. The model is running up to **86% faster** than models with comparable performance when encoding queries. That efficiency unlocks new use cases for visual retrieval and closes the model size gap to popular text retrievers, which in practice are often used with CPUs.

## Links
- Paper 📄: https://arxiv.org/abs/2510.01149
- HF Org 🤗: https://huggingface.co/ModernVBERT
- Codebase 💻: https://github.com/paultltc/modernvbert
- Finetuning Tutorial 🧑‍🍳: https://colab.research.google.com/drive/1bT5LWeO1gPL83GKUZsFeFEleHmEDEQRy

## Authors
- Paul Teiletche (✖️@pteiletche)
- Quentin Macé (✖️@MaceQuent1)
- Max Conti (✖️@mlpc123)
- Antonio Loison (✖️@antonio_loison)
- Manuel Faysse (✖️@ManuelFaysse)

## Citation
You can cite us in the following way: 
```bibtex
@misc{teiletche2025modernvbertsmallervisualdocument,
      title={ModernVBERT: Towards Smaller Visual Document Retrievers}, 
      author={Paul Teiletche and Quentin Macé and Max Conti and Antonio Loison and Gautier Viaud and Pierre Colombo and Manuel Faysse},
      year={2025},
      eprint={2510.01149},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2510.01149}, 
}
```

## Acknowledgments
This work was carried out within the framework of the LIAGORA ”LabCom”, a joint laboratory supported by the French National Research Agency (ANR) and established between ILLUIN Tech- nology and the MICS laboratory of CentraleSupélec. This work was performed using HPC resources from IDRIS with grant AD011016393. We warmly thank Hippolyte Gisserot-Boukhlef and Nicolas Boizard for sharing the controlled experiments LM checkpoints, Antoine Chaffin for his feedback on the modality alignment codebase and insights on Ettin’s modeling, as well as the SmolVLM team for their valuable input and help on gathering the modality alignment dataset. 		