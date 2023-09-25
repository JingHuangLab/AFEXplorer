# A RunModel wrapper customized for AFExplore.
# Zilin Song, 20230818
# 

"""AFExplore RunModel."""

from typing import Any, Mapping, Optional

import numpy as np, jax, haiku as hk

from ml_collections import ConfigDict

from alphafold.model.features import FeatureDict

from alphafold.model import config, data, model, modules

class AFExploreRunModel(model.RunModel):
  """AFExplore customized container for AF2 JAX model."""

  def __init__(self, 
               config: ConfigDict, 
               params: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None, ):
    """Create a AFExplore customized container for AF2 Jax model.
    
      Args:
        config: The configuration dicts.
        params: The model parameters.
    """
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode # Always False -> Monomer only.

    # Removed multimer_mode _forward_fn
    def _forward_fn(batch):
      """AF forward pass."""
      m = modules.AlphaFold(self.config.model)

      # ./alphafold/model/tf/data_transforms.py
      #
      # msa_feat = [
      #     msa_1hot,                                           <-  0:23
      #     tf.expand_dims(has_deletion, axis=-1),              <- 23:24
      #     tf.expand_dims(deletion_value, axis=-1),            <- 24:25
      # ]
      # if 'cluster_profile' in protein: # This is always true.
      #   deletion_mean_value = (
      #       tf.atan(protein['cluster_deletion_mean'] / 3.) * (2. / np.pi))
      #   msa_feat.extend([
      #       protein['cluster_profile'],                       <- 25:48
      #       tf.expand_dims(deletion_mean_value, axis=-1),     <- 48:49
      #   ])
      #

      batch['msa_feat'] = batch['msa_feat'].at[:, :, :, 25:48].set(
        batch['msa_feat'][:, :, :, 25:48]*batch['afe_msa_feat_weights'] + 
        batch['afe_msa_feat_bias'])
      
      return m(batch,
               is_training=False,
               compute_loss=False,
               ensemble_representations=False, )

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init  = jax.jit(hk.transform(_forward_fn).init )
  
  def predict(self, 
              feat: FeatureDict, 
              afe_params: tuple, 
              random_seed: int, 
              ) -> Mapping[str, Any]:
    """Makes a prediction by inferencing the model on the provided features.

      Args:
        feat: A dictionary of NumPy feature arrays as output by
          RunModel.process_features.
        random_seed: The random seed to use when running the model. In the
          multimer model this controls the MSA sampling.
          
      Returns:
        A dictionary of model outputs, without the metrics.
    """
    feat["afe_msa_feat_weights"], feat["afe_msa_feat_bias"]  = afe_params
    # self.init_params(feat=feat)
    # logging.info('Running predict with shape(feat) = %s', tree.map_structure(lambda x: x.shape, feat))
    result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)
    return result

def get_afe_runner(afparam_dir: str, 
                   model_name:  str, 
                   num_cluster: int = 512, # This is the default value.
                   num_recycle: int = 0, 
                   recycle_feat: bool = True, 
                   recycle_pos:  bool = True, 
                   resample_msa_in_recycling: bool = False, 
                   deterministic: bool = True, 
                   ) -> AFExploreRunModel:
  """Get a configured AFExploreRunModel with no template."""
  model_config = config.model_config(model_name)

  # Turn off multimer_model.
  model_config.model.global_config.multimer_mode = False

  # Configs for feature pipeline (config.data).
  ## num_ensemble in data pipeline: 
  ##   config.data.eval.num_ensemble:  The number of msa ensembles in each recycle.
  ##   config.data.common.num_recycle: The number of msa recycling in the embending&evoformer pass.
  ##
  ## alphafold.model.tf.input_pipeline.process_tensors_from_config():
  ##   num_ensemble/batch_size = config.data.eval.num_ensemble * (config.data.common.num_recycle + 1)
  ## This config is used in RunModel.process_features().
  model_config.data.common.num_recycle = num_recycle
  model_config.data.eval.num_ensemble  = 1


  model_config.data.eval.max_msa_clusters = num_cluster
  
  ## Used at alphafold.features.make_data_config().
  ## Used at alphafold.model.input_pipeline.nonensembled_map_fns().
  model_config.data.common.use_templates = False
  ## Used at alphafold.model.input_pipeline.ensembled_map_fns().
  ## Determines: config.data.eval.max_msa_clusters & config.data.common.max_extra_msa.
  model_config.data.common.reduce_msa_clusters_by_max_templates = False
  model_config.data.eval.max_templates = 0

  # Configs for model forward pass (config.model).
  ## num_ensemble in model inference:
  ##   num_ensemble(_per_batch): The number of ensembles for the current recycle pass.
  ##   num_ensemble/batch_size:  The number of all ensembles through all recycle passes.
  ##   config.model.num_recycle: The number of msa recycling in the embedding&evoformer pass.
  ## 
  ## alphafold.modules.AlphaFold.__call__.do_call()
  ##   num_ensemble(_per_batch) = num_ensemble/batch_size // (config.model.num_recycle + 1)
  ## This config is used in RunModel.predict(), must be <= to config.data.common.num_recycle.
  model_config.model.num_recycle = num_recycle
  ## Used at alphafold.modules.AlphaFold().
  model_config.model.resample_msa_in_recycling = resample_msa_in_recycling
  # ## Used at alphafold.modules.AlphaFold() - for init first recycling.
  # ##         alphafold.modules.EmbeddingsAndEvoformer()
  model_config.model.embeddings_and_evoformer.recycle_features = recycle_feat
  model_config.model.embeddings_and_evoformer.recycle_pos      = recycle_pos

  ## Used at alphafold.modules.EmbeddingsAndEvoformer()
  model_config.model.embeddings_and_evoformer.template.enabled = False
  
  ## Used in all dropouts: alphafold.modules.dropout_wrapper()
  model_config.model.global_config.deterministic = deterministic

  ## To fully remove stochasity in AF2, if recycling is not required:  
  ## Set 1) config.model.num_recycle == 0 and 2) model.global_config.deterministic == True

  ## AF2 graph in reverse mode.
  model_config.model.global_config.use_remat = True

  # Runner: params.
  model_params = data.get_model_haiku_params(model_name=model_name, data_dir=afparam_dir)

  # Runner: make.
  model_runner = AFExploreRunModel(config=model_config, params=model_params)

  return model_runner