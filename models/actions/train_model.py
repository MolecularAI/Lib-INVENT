import torch.nn.utils as tnnu
import torch.utils.data as tud

import models.dataset as md
from models.actions import Action
from models.actions.collect_stats_from_model import CollectStatsFromModel
from running_modes.configurations import TransferLearningConfiguration
from running_modes.enums import GenerativeModelRegimeEnum
from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger


class TrainModel(Action):

    def __init__(self, model, configuration: TransferLearningConfiguration, optimizer, training_sets, validation_sets,
                 lr_scheduler, logger: BaseTransferLearningLogger):
        """
        Initializes the training of an epoch.
        : param model: A model instance, not loaded in scaffold_decorating mode.
        : param optimizer: The optimizer instance already initialized on the model.
        : param training_sets: An iterator with all the training sets (scaffold, decoration) pairs.
        : param batch_size: Batch size to use.
        : param clip_gradient: Clip the gradients after each backpropagation.
        : return:
        """
        Action.__init__(self, logger)

        self.model = model
        self.config = configuration
        self.optimizer = optimizer
        self.training_sets = training_sets
        self.validation_sets = validation_sets
        self.lr_scheduler = lr_scheduler
        self.model_regime_enum = GenerativeModelRegimeEnum()

    def run(self):
        for epoch, training_set, validation_set in zip(range(1, self.config.epochs + 1), self.training_sets,
                                                       self.validation_sets):
            dataloader = self._initialize_dataloader(training_set)

            # iterate over training batch
            for scaffold_batch, decorator_batch in dataloader:
                loss = self.model.likelihood(*scaffold_batch, *decorator_batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.clip_gradients > 0:
                    tnnu.clip_grad_norm_(self.model.network.parameters(), self.config.clip_gradients)

                self.optimizer.step()

            # Get stats and logs
            self.collect_stats(epoch=epoch, training_set=training_set, validation_set=validation_set)

            # update LR
            self.lr_scheduler.step()

            # determine if training should continue
            terminate = self.checkpoint(self.lr_scheduler.optimizer.param_groups[0]["lr"], epoch)

            if terminate:
                self.model.save(f"{self.config.output_path}/trained.{epoch}")
                break

    def collect_stats(self, epoch, training_set, validation_set):
        self.model.set_mode(self.model_regime_enum.INFERENCE)
        stats = CollectStatsFromModel(model=self.model, epoch=epoch, sample_size=self.config.sample_size,
                                      training_set=training_set, validation_set=validation_set).run()

        self.logger.log_timestep(lr=self.lr_scheduler.optimizer.param_groups[0]["lr"], epoch=epoch,
                                 training_smiles=stats['sampled_training_mols'],
                                 validation_smiles=stats['sampled_validation_mols'],
                                 validation_nlls=stats['validation_nlls'],
                                 training_nlls=stats['training_nlls'],
                                 jsd_data_no_bins=stats['unbinned_jsd'],
                                 jsd_data_bins=stats['binned_jsd'],
                                 model=self.model)
        self.model.set_mode(self.model_regime_enum.TRAINING)

    def checkpoint(self, lr, epoch):
        terminate_flag = False
        if lr < self.config.learning_rate.min:
            self.logger.log_message("Reached LR minimum. Terminating.")
            terminate_flag = True

        elif self.config.epochs == epoch:
            self.logger.log_message(f"Reached maximum number of epochs ({epoch}). Saving and terminating.")
            terminate_flag = True

        elif self.config.save_frequency > 0 and (epoch % self.config.save_frequency == 0):
            self.model.save(f"{self.config.output_path}/trained.{epoch}")
            self.logger.log_message(f"Checkpoint after epoch {epoch}. Saving the model.")
        return terminate_flag

    def _initialize_dataloader(self, training_set):
        dataset = md.DecoratorDataset(training_set, vocabulary=self.model.vocabulary)
        return tud.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True,
                              collate_fn=md.DecoratorDataset.collate_fn, drop_last=True)
