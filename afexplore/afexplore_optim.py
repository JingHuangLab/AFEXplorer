# Make prediction with AF2.
# Zilin Song, 20230820
# Tengyu Xie, 20230913

import os

import sys

import numpy as np

from typing import Tuple

import jax, jax.numpy as jnp, optax

from absl import app, flags

from alphafold.model import config

from alphafold.common import protein

from afexplore_runner import get_afe_runner, AFExploreRunModel

import pandas as pd

# DIR: raw_features as input.
flags.DEFINE_string('rawfeat_dir', None, 'Path to directory that stores the raw features.')

# DIR: output.
flags.DEFINE_string('output_dir', None, 'Path to a directory that stores all outputs.')

# DIR: data.
flags.DEFINE_string('afparam_dir', None, 'Path to directory of supporting data / model parameters.')

# Config: number of optimization steps.
flags.DEFINE_integer('nsteps', 10, 'Number of optimization steps.')
# Config-AF: number of MSA clusters
flags.DEFINE_integer('nclust', 512, 'Number of MSA clusters used for featurization, this number scales linearly with memory usage.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for gradient updates.')

# PRESET: models: monomer only.
flags.DEFINE_enum('model_preset', 'monomer', ['monomer', ],   # 'monomer_casp14', 'monomer_ptm', 'multimer'
                  'Choose preset model configuration - the monomer model, the monomer model with extra ensembling, monomer model with pTM head, or multimer model')

flags.DEFINE_enum('protein_type', 'kinase',
                  ['kinase', 'IOMemP', 'ADK'],
                  'Choose protein type - kinase, membrane protein IOMemP and ADK. '
                  'Loss functions of other proteins are required to be added accordingly.')
flags.DEFINE_enum('target_state', 'in',
                  ['in', 'out'],
                  'Alternative states of kinase/membrane proteins(IOMemP)/ADK:'
                  'in refers to DFGin/closed/inward-opening; '
                  'out refers to DFGout/open/outward-facing.')
flags.DEFINE_integer('num_success', None, 'The number of accumalated successful samplings.')
flags.DEFINE_float('cutoff', 10, 'The cutoff distance for CV for IOMemP.')

FLAGS = flags.FLAGS

def afe_fitting(afe_runner: AFExploreRunModel,
                af_features: dict, n_steps: int, 
                learning_rate: float,
                ) -> optax.Params:
  """Fit the AFExplore model."""
  afe_weights = jnp.ones(( af_features['msa_feat'].shape[0], 
                           af_features['msa_feat'].shape[1], 
                           af_features['msa_feat'].shape[2], 
                           23, ), )
  afe_bias = jnp.zeros(( af_features['msa_feat'].shape[0], 
                          af_features['msa_feat'].shape[1], 
                          af_features['msa_feat'].shape[2], 
                          23, ), )


  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params=(afe_weights, afe_bias))
  
  # ------------------------------------------------------------------------------------------------
  # Prepare for loss function.
  if FLAGS.protein_type == 'kinase':
    # Import Kincore only for kinase.
    from Kincore import kinase_state
    # from Kincore import modules

    # recognize residue indices for CV by Kincore
    dfg_AFmodel1_path = os.path.join(FLAGS.output_dir, 'unrelaxed_model_1_pred_0.pdb') # unrelaxed_model_1_pred_0.pdb
    hmm_loc = os.path.dirname(os.path.realpath(__file__))+'/Kincore/HMMs'    #gets the original location of the file
    conf_df = kinase_state.identify_state(hmm_loc, dfg_AFmodel1_path, 'True', '', '', '', '', '') 
    state_afmodel1 = conf_df.at[0,'Spatial_label'] # DFGout or DFGin
    phe_glu_plus4_lys_pre = np.asarray([conf_df.at[0,'Phe_num'], 
                                        conf_df.at[0,'Glu4_num'], 
                                        conf_df.at[0,'Lys_num']], dtype='int16') 
    
    phe_glu_plus4_lys_pre_offset = phe_glu_plus4_lys_pre - 1
    
    print("State of AFmodel1: ", state_afmodel1)
    print("Indices of Phe, Glu+4, Lys: ", phe_glu_plus4_lys_pre) 
    
    sys.stdout.flush()

  elif FLAGS.protein_type == 'IOMemP':
    # read indices of two residues for CV
    gate_res_path = os.path.join(FLAGS.output_dir, 'gate_residues')
    with open(gate_res_path, 'r') as f:
      reses = []
      for line in f:
        res = int(line[1:-1])
        reses.append(res)
    
    res1, res2 = reses
  # ------------------------------------------------------------------------------------------------
  def afe_loss_fn(afe_params: Tuple, 
                  af_features: dict, 
                  ) -> Tuple[jnp.ndarray, dict]:
    prediction_result = afe_runner.predict(af_features, afe_params, 0)

    # Loss from pLDDT
    # c.f. ./alphafold/common/confidence.py
    plddt_logits = prediction_result['predicted_lddt']['logits']
    plddt_bin_width = 1./plddt_logits.shape[-1]
    plddt_bin_centers = jnp.arange(start=.5*plddt_bin_width, stop=1., step=plddt_bin_width, )
    plddt_ca = jnp.sum(jax.nn.softmax(plddt_logits, axis=-1)*plddt_bin_centers[None, :], axis=-1)
    plddt_loss = 1.-jnp.mean(plddt_ca)
    

    # Loss from CV.
    if FLAGS.protein_type == 'kinase':
      phe_CZ = prediction_result['structure_module']['final_atom_positions'][phe_glu_plus4_lys_pre_offset[0], -5, :]
      glu_plus4_CA = prediction_result['structure_module']['final_atom_positions'][phe_glu_plus4_lys_pre_offset[1], 1, :]
      lys_CA = prediction_result['structure_module']['final_atom_positions'][phe_glu_plus4_lys_pre_offset[2], 1, :]
      
      d1 = jnp.sqrt(jnp.sum((phe_CZ - glu_plus4_CA) ** 2))
      d2 = jnp.sqrt(jnp.sum((phe_CZ - lys_CA) ** 2))
      
      if FLAGS.target_state == 'in': # for DFGin
        # d1<=11 and d2>=11
        d1_loss = -jax.nn.log_sigmoid(-(d1 - 11))
        d2_loss = -jax.nn.log_sigmoid(d2 - 11)
      elif FLAGS.target_state == 'out': # for DFGout
        # d1>11 and d2<=14
        d1_loss = -jax.nn.log_sigmoid(d1 - 11)
        d2_loss = -jax.nn.log_sigmoid(-(d2 - 14))

      return plddt_loss+d1_loss+d2_loss, (prediction_result, 
                                             jax.lax.stop_gradient(plddt_loss), 
                                             jax.lax.stop_gradient(d1_loss),
                                             jax.lax.stop_gradient(d2_loss), 
                                             jax.lax.stop_gradient(d1), 
                                             jax.lax.stop_gradient(d2) )
    elif FLAGS.protein_type == 'ADK':
      A37_CA = prediction_result['structure_module']['final_atom_positions'][36, 1, :]
      R124_CA = prediction_result['structure_module']['final_atom_positions'][123, 1, :]
      d1 = jnp.sqrt(jnp.sum((A37_CA - R124_CA) ** 2))
                
      if FLAGS.target_state == 'in': # for closed state
        # d1 <= 25
        d1_loss = -jax.nn.log_sigmoid(-(d1 - 25))
      
      elif FLAGS.target_state == 'out': # for open state
        # d1 >= 25
        d1_loss = -jax.nn.log_sigmoid(d1 - 25)

      return plddt_loss+d1_loss, (prediction_result, 
                                  jax.lax.stop_gradient(plddt_loss), 
                                  jax.lax.stop_gradient(d1_loss), 
                                  jax.lax.stop_gradient(d1))
    elif FLAGS.protein_type == 'IOMemP':
      res1_CA = prediction_result['structure_module']['final_atom_positions'][res1-1, 1, :]
      res2_CA = prediction_result['structure_module']['final_atom_positions'][res2-1, 1, :]
      d1 = jnp.sqrt(jnp.sum((res1_CA - res2_CA) ** 2))

      if FLAGS.target_state == 'in': # for IF state
        # d1 <= cutoff
        d1_loss = -jax.nn.log_sigmoid(-(d1 - FLAGS.cutoff))
      
      elif FLAGS.target_state == 'out': # for OF state
        # d1 >= cufoff
        d1_loss = -jax.nn.log_sigmoid(d1 - FLAGS.cutoff)

      return plddt_loss+d1_loss, (prediction_result, 
                                  jax.lax.stop_gradient(plddt_loss), 
                                  jax.lax.stop_gradient(d1_loss), 
                                  jax.lax.stop_gradient(d1))
  # ------------------------------------------------------------------------------------------------

  count = 0
  losses = []
  plddts = []
  d1_losses = []
  d2_losses = []
  d1s = []
  d2s = []

  for i in range(n_steps): # One optimization step.

    (loss, aux), grads = jax.value_and_grad(afe_loss_fn, has_aux=True)((afe_weights, afe_bias), af_features)
    updates, opt_state = optimizer.update(updates=grads,  state=opt_state)
    afe_weights, afe_bias = optax.apply_updates(params=(afe_weights, afe_bias), updates=updates)

    if FLAGS.protein_type == 'kinase':
      # Check state
      if FLAGS.target_state == 'in': # for DFGin
        # d1<=11 and d2>=11
        if aux[4]<=11 and aux[5]>=11:
          count += 1
          print('D1 and D2 threshold reached.')

      elif FLAGS.target_state == 'out': # for DFGout
        # d1>11 and d2<=14      
        if aux[4]>11 and aux[5]<=14:
          count += 1
          print('D1 and D2 threshold reached.')
      
      if count >= FLAGS.num_success:
        print('DFGin/DFGout threshold reached for %s times.'%(FLAGS.num_success))
        break

      losses.append(loss)
      plddts.append(aux[1])
      d1_losses.append(aux[2])
      d2_losses.append(aux[3])
      d1s.append(aux[4])
      d2s.append(aux[5])
      
      print('Step:', i+1, '|loss:', loss, '|best loss:', np.min(losses),
            '|plddt:', aux[1], '|best plddt:', np.min(plddts),
            '|d1_loss:', aux[2], '|best d1_loss:', np.min(d1_losses),
            '|d2_loss:', aux[3], '|best d2_loss:', np.min(d2_losses),
            '|d1:', aux[4], 
            '|d2:', aux[5]) 
      
    elif FLAGS.protein_type == 'ADK' or FLAGS.protein_type == 'IOMemP':
      losses.append(loss)
      plddts.append(aux[1])
      d1_losses.append(aux[2])
      d1s.append(aux[3])
      
      print('Step:', i+1, '|loss:', loss, '|best loss:', np.min(losses),
            '|plddt:', aux[1], '|best plddt:', np.min(plddts),
            '|d1_loss:', aux[2], '|best d1_loss:', np.min(d1_losses),
            '|d1:', aux[3])
    
    sys.stdout.flush()
            
    p = protein.from_prediction(features=af_features, 
                                result=aux[0], 
                                b_factors=None, 
                                remove_leading_feature_dimension=True, )  # True for Monomer.
    
    with open(os.path.join(FLAGS.output_dir, f'afe_model_{i}.pdb'), 'w') as f:
      f.write(protein.to_pdb(p))
  
  if FLAGS.protein_type == 'kinase':
    metric_df = pd.DataFrame({'loss': losses, 'plddt': plddts, 'd1_loss': d1_losses, 'd2_loss': d2_losses, 'd1': d1s, 'd2': d2s})
  elif FLAGS.protein_type == 'ADK' or FLAGS.protein_type == 'IOMemP':
    metric_df = pd.DataFrame({'loss': losses, 'plddt': plddts, 'd1_loss': d1_losses, 'd1': d1s})
  
  metric_df.to_csv(FLAGS.output_dir+'/metrics_'+str(FLAGS.learning_rate)+'.csv', index=None) 

def main(argv):
  """AF2 inference."""
  # Sanity checks: Args count.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  model_runner = get_afe_runner(afparam_dir=FLAGS.afparam_dir, 
                                model_name=config.MODEL_PRESETS[FLAGS.model_preset][0], 
                                num_cluster=FLAGS.nclust, )
  
  # Load featurized MSAs.
  raw_feat = np.load(os.path.join(FLAGS.rawfeat_dir, 'features.pkl'), allow_pickle=True)

  # Process to real features.
  feat = model_runner.process_features(raw_features=raw_feat, 
                                       random_seed=123, )

  # feat = jnp.load(os.path.join(FLAGS.rawfeat_dir, 'processed_features.pkl'), allow_pickle=True)

  afe_fitting(afe_runner=model_runner,
              af_features=feat, 
              n_steps=FLAGS.nsteps, 
              learning_rate=FLAGS.learning_rate)
  
if __name__ == '__main__':
  flags.mark_flags_as_required([
      'rawfeat_dir',
      'output_dir',
      'afparam_dir',
      'nsteps', 
      'nclust'
  ])

  app.run(main)
