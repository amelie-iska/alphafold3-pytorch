<img src="./alphafold3.png" width="450px"></img>

## Alphafold 3 - Pytorch (wip)

Implementation of <a href="https://www.nature.com/articles/s41586-024-07487-w">Alphafold 3</a> in Pytorch

Getting a fair number of emails. You can chat with me about this work <a href="https://discord.gg/x6FuzQPQXY">here</a>

## Appreciation

- <a href="https://github.com/joseph-c-kim">Joseph</a> for contributing the Relative Positional Encoding and the Smooth LDDT Loss!

- <a href="https://github.com/engelberger">Felipe</a> for contributing Weighted Rigid Align, Express Coordinates In Frame, Compute Alignment Error, and Centre Random Augmentation modules!

## Install

```bash
$ pip install alphafold3-pytorch
```

## Usage

```python
import torch
from alphafold3_pytorch import Alphafold3

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_additional_residue_feats = 33,
    dim_template_feats = 44
)

# mock inputs

seq_len = 16
atom_seq_len = seq_len * 27

atom_inputs = torch.randn(2, atom_seq_len, 77)
atom_mask = torch.ones((2, atom_seq_len)).bool()
atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
additional_residue_feats = torch.randn(2, seq_len, 33)

template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 64)

# required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)
residue_atom_indices = torch.randint(0, 27, (2, seq_len))

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
pae_labels = torch.randint(0, 64, (2, seq_len, seq_len))
pde_labels = torch.randint(0, 64, (2, seq_len, seq_len))
plddt_labels = torch.randint(0, 50, (2, seq_len))
resolved_labels = torch.randint(0, 2, (2, seq_len))

# train

loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    atom_mask = atom_mask,
    atompair_feats = atompair_feats,
    additional_residue_feats = additional_residue_feats,
    msa = msa,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    residue_atom_indices = residue_atom_indices,
    distance_labels = distance_labels,
    pae_labels = pae_labels,
    pde_labels = pde_labels,
    plddt_labels = plddt_labels,
    resolved_labels = resolved_labels
)

loss.backward()

# after much training ...

sampled_atom_pos = alphafold3(
    num_recycling_steps = 4,
    num_sample_steps = 16,
    atom_inputs = atom_inputs,
    atom_mask = atom_mask,
    atompair_feats = atompair_feats,
    additional_residue_feats = additional_residue_feats,
    msa = msa,
    templates = template_feats,
    template_mask = template_mask
)

sampled_atom_pos.shape # (2, 16 * 27, 3)
```

## Flow Matching Usage

First, initialize the `AlphaFold3` class with the necessary parameters:

```python
# Example initialization with dummy parameters
alphafold3 = AlphaFold3(
    dim_atom_inputs=64,
    dim_additional_residue_feats=10,
    dim_template_feats=64,
    atoms_per_window=27,
    dim_atom=128,
    dim_atompair=16,
    dim_input_embedder_token=384,
    dim_single=384,
    dim_pairwise=128,
    dim_token=768,
    atompair_dist_bins=torch.linspace(3, 20, 37),
    ignore_index=-1,
    num_dist_bins=38,
    num_plddt_bins=50,
    num_pde_bins=64,
    num_pae_bins=64,
    sigma_data=16,
    flow_matching_num_augmentations=4,
    loss_confidence_weight=1e-4,
    loss_distogram_weight=1e-2,
    loss_flow_matching_weight=4.,
    input_embedder_kwargs=dict(
        atom_transformer_blocks=3,
        atom_transformer_heads=4,
        atom_transformer_kwargs=dict()
    ),
    confidence_head_kwargs=dict(
        pairformer_depth=4
    ),
    template_embedder_kwargs=dict(
        pairformer_stack_depth=2,
        pairwise_block_kwargs=dict(),
    ),
    msa_module_kwargs=dict(
        depth=4,
        dim_msa=64,
        dim_msa_input=None,
        outer_product_mean_dim_hidden=32,
        msa_pwa_dropout_row_prob=0.15,
        msa_pwa_heads=8,
        msa_pwa_dim_head=32,
        pairwise_block_kwargs=dict()
    ),
    pairformer_stack=dict(
        depth=48,
        pair_bias_attn_dim_head=64,
        pair_bias_attn_heads=16,
        dropout_row_prob=0.25,
        pairwise_block_kwargs=dict()
    ),
    relative_position_encoding_kwargs=dict(
        r_max=32,
        s_max=2,
    ),
    flow_matching_module_kwargs=dict(
        single_cond_kwargs=dict(
            num_transitions=2,
            transition_expansion_factor=2,
        ),
        pairwise_cond_kwargs=dict(
            num_transitions=2
        ),
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=24,
        token_transformer_heads=16,
        atom_decoder_depth=3,
        atom_decoder_heads=4
    ),
    flow_matching_kwargs=dict(
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
        S_churn=80,
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
    ),
    augment_kwargs=dict()
)
```

To perform a forward pass and calculate the loss:

```python
# Dummy data for demonstration
batch_size = 4
num_atoms = 27
num_residues = 64

atom_inputs = torch.randn(batch_size, num_atoms, 64)
atom_mask = torch.ones(batch_size, num_atoms).bool()
atompair_feats = torch.randn(batch_size, num_atoms, num_atoms, 16)
additional_residue_feats = torch.randn(batch_size, num_residues, 10)
msa = torch.randn(batch_size, 128, num_residues, 64)
msa_mask = torch.ones(batch_size, 128).bool()
templates = torch.randn(batch_size, 8, num_residues, num_residues, 64)
template_mask = torch.ones(batch_size, 8).bool()
atom_pos = torch.randn(batch_size, num_atoms, 3)
distance_labels = torch.randint(0, 38, (batch_size, num_residues, num_residues))
pae_labels = torch.randint(0, 64, (batch_size, num_residues, num_residues))
pde_labels = torch.randint(0, 64, (batch_size, num_residues, num_residues))
plddt_labels = torch.randint(0, 50, (batch_size, num_residues))
resolved_labels = torch.randint(0, 2, (batch_size, num_residues))

loss = alphafold3(
    atom_inputs=atom_inputs,
    atom_mask=atom_mask,
    atompair_feats=atompair_feats,
    additional_residue_feats=additional_residue_feats,
    msa=msa,
    msa_mask=msa_mask,
    templates=templates,
    template_mask=template_mask,
    num_recycling_steps=1,
    flow_matching_add_bond_loss=False,
    flow_matching_add_smooth_lddt_loss=False,
    residue_atom_indices=None,
    num_sample_steps=None,
    atom_pos=atom_pos,
    distance_labels=distance_labels,
    pae_labels=pae_labels,
    pde_labels=pde_labels,
    plddt_labels=plddt_labels,
    resolved_labels=resolved_labels,
    return_loss_breakdown=False
)

print("Loss:", loss)
```

## Contributing

At the project root, run

```bash
$ sh ./contribute.sh
```

Then, add your module to `alphafold3_pytorch/alphafold3.py`, add your tests to `tests/test_af3.py`, and submit a pull request. You can run the tests locally with

```bash
$ pytest tests/
```

## Citations

```bibtex
@article{Abramson2024-fj,
  title    = "Accurate structure prediction of biomolecular interactions with
              {AlphaFold} 3",
  author   = "Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans,
              Richard and Green, Tim and Pritzel, Alexander and Ronneberger,
              Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick,
              Joshua and Bodenstein, Sebastian W and Evans, David A and Hung,
              Chia-Chun and O'Neill, Michael and Reiman, David and
              Tunyasuvunakool, Kathryn and Wu, Zachary and {\v Z}emgulyt{\.e},
              Akvil{\.e} and Arvaniti, Eirini and Beattie, Charles and
              Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and
              Congreve, Miles and Cowen-Rivers, Alexander I and Cowie, Andrew
              and Figurnov, Michael and Fuchs, Fabian B and Gladman, Hannah and
              Jain, Rishub and Khan, Yousuf A and Low, Caroline M R and Perlin,
              Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and
              Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine
              and Yakneen, Sergei and Zhong, Ellen D and Zielinski, Michal and
              {\v Z}{\'\i}dek, Augustin and Bapst, Victor and Kohli, Pushmeet
              and Jaderberg, Max and Hassabis, Demis and Jumper, John M",
  journal  = "Nature",
  month    = "May",
  year     =  2024
}
```
