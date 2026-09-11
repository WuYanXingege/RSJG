import os
import time
import numpy as np
from tqdm import tqdm
import torch
from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.model_utils.U_net_CNN import UNet
from src.losses import Goal_BCE_loss
from src.utils import add_dict_prefix, formatted_time, print_model_summary, find_trainable_layers
from src.data_loader import get_dataloader
from src.metrics import compute_metric_mask, ADE_best_of, FDE_best_of
from src.models.model_utils.sampling_2D_map import TTST_test_time_sampling_trick

def derivative_of(input, dt=1, radian=False):
    '''
    input: (T,)
    output: (T,)
    '''
    device = input.device
    x = input.clone().detach().cpu()
    x = x.numpy()
    dx = np.full_like(x, np.nan)
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt) # (T-2, 2)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    dx = torch.from_numpy(dx).float().to(device)
    return dx

# def derivative_of(input, dt=1, radian=False):
#     '''
#     input: (batch_size, T, 2)
#     output: (batch_size, T, 2)
#     '''
#     batch_size, T, _ = input.shape
#     device = input.device
#     x = input.clone().detach().cpu()
#     x = x.numpy()
#     dx = np.full_like(x, np.nan)
#     for i in range(batch_size):
#         x_i = x[i] # (T, 2)
#         dx_i = np.zeros_like(x_i)
#         dx_i[1:-1] = (x_i[2:] - x_i[:-2]) / (2 * dt) # (T-2, 2)
#         dx_i[0] = (x_i[1] - x_i[0]) / dt
#         dx_i[-1] = (x_i[-1] - x_i[-2]) / dt
#         # dx_i = np.full_like(x_i, np.nan)
#         # dx_i[~np.isnan(x_i)] = np.gradient(x_i[~np.isnan(x_i)], dt)
#         dx[i] = dx_i
#     dx = torch.from_numpy(dx).float().to(device)
#     return dx

class Goal_Pretrain(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dataset = create_dataset(
            self.args.dataset, load_visual_data=False)

        ##################
        # MODEL PARAMETERS
        ##################
        self.output_size = 2
        # GOAL MODULE PARAMETERS
        self.num_image_channels = 6

        # U-net encoder channels
        self.enc_chs = (self.num_image_channels + self.args.obs_length,
                        32, 32, 64, 64, 64)
        # U-net decoder channels
        self.dec_chs = (64, 64, 64, 32, 32)
        ##################
        # MODEL LAYERS
        ##################
        self.goal_module = UNet(
            enc_chs=self.enc_chs,
            dec_chs=self.dec_chs,
            out_chs=self.args.pred_length)

    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float
        selected_inputs = {k: v.squeeze(0).float().to(self.device) if \
            torch.is_tensor(v) else v for k, v in batch_data.items()}
        # extract seq_list
        seq_list = selected_inputs["seq_list"]
        # decide which is ground truth
        ground_truth = selected_inputs["abs_pixel_coord"] # T,B,2

        scene_name = batch_id["scene_name"][0]
        scene = self.dataset.scenes[scene_name]
        selected_inputs["scene"] = scene

        return selected_inputs, ground_truth.detach(), seq_list.detach()

    def init_losses(self):
        losses = {
            "goal_BCE_loss": 0,
        }
        return losses

    def set_losses_coeffs(self):
        losses_coeffs = {
            "goal_BCE_loss": 1,
        } 
        return losses_coeffs
    
    
    def init_train_metrics(self):
        train_metrics = {
            "goal_BCE": [],
            'ADE_world_traj': [],
            'FDE_world_traj': [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "goal_BCE": [],
            'ADE_world_traj': [],
            'FDE_world_traj': [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "goal_BCE": 1e9,
            'ADE_world_traj': 1e9,
            'FDE_world_traj': 1e9,
        }
        return best_metrics

    def best_valid_metric(self):
        return "FDE_world_traj" 

    def compute_model_metrics(self,
                              metric_name,
                              predictions,
                              ground_truth,
                              seq_list,
                              metric_mask,
                              inputs,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """
        # scale back to original dimension
        predictions = predictions.detach() * self.args.down_factor
        ground_truth = ground_truth.detach() * self.args.down_factor
        # convert to world coordinates
        scene = inputs["scene"]

        GT_world = scene.make_world_coord_torch(ground_truth)

        
        if metric_name == 'goal_BCE':
            # compute goal loss
            loss_mask = self.compute_loss_mask(
                seq_list, self.args.obs_length).to(self.device)
            out_maps_GT_goal = inputs["input_traj_maps"][:, self.args.obs_length:]
            goal_logit_map = predictions
            goal_BCE = Goal_BCE_loss(
                goal_logit_map, out_maps_GT_goal, loss_mask)
            return [goal_BCE.tolist()]
        else:
            raise ValueError("This metric has not been implemented yet!")

    def goal_logit_map_prediction(self, inputs):
        obs_traj_maps = inputs["input_traj_maps"][:, 0:self.args.obs_length]
        num_agents = obs_traj_maps.shape[0]
        # extract precomputed map for goal goal_idx
        tensor_image = inputs["tensor_image"].unsqueeze(0).repeat(num_agents, 1, 1, 1) # add the first dim, repeat [1, C, H, W] to [num_agents, C, H, W]
        input_goal_module = torch.cat((tensor_image, obs_traj_maps), dim=1)
        # compute goal maps, UNet as backbone
        goal_logit_map_start = self.goal_module(input_goal_module) # (num_agents, C+T, H, W) -> (num_agents, C_out, H, W), C_out = 12
        return goal_logit_map_start.unsqueeze(0) # (1, num_agents, C_out, H, W)

    def forward(self, inputs):
        goal_logit_map_start = self.goal_logit_map_prediction(inputs).squeeze(0)
        goal_prob_map = torch.sigmoid(goal_logit_map_start[:, -1:]) # select only the last column of second dim and take sigmoid,(num_agents, 1, H, W)
        goal_point_start = TTST_test_time_sampling_trick(
            goal_prob_map,
            num_goals=20,
            device=self.device)
        goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2) # final result: (num_agents, num_samples, 2)
        batch_coords = inputs["abs_pixel_coord"].detach()
        # Number of agent in current batch_abs_world
        seq_length, num_agents, _ = batch_coords.shape # (T, B, 2)

        return goal_logit_map_start.unsqueeze(0)


    def get_loss(self, inputs, seq_list):
        goal_logit_map = self.goal_logit_map_prediction(inputs) 

        # compute loss_mask
        loss_mask = self.compute_loss_mask(
            seq_list, self.args.obs_length).to(self.device)

        # compute goal loss
        out_maps_GT_goal = inputs["input_traj_maps"][:, self.args.obs_length:]
        goal_BCE_loss = Goal_BCE_loss(goal_logit_map, out_maps_GT_goal, loss_mask)
        losses = {"goal_BCE_loss": goal_BCE_loss}

        return losses
    

class goal_pretrainer(object):
    def __init__(self, args):
        self.args = args
        # initialize data loaders
        self.data_loaders = dict()
        self.data_loaders['train'] = get_dataloader(args, set_name='train')
        self.data_loaders['valid'] = get_dataloader(args, set_name='valid')

        # initialize device
        self.device = self._set_device()
        # initialize network
        self.net = Goal_Pretrain(self.args, self.device).to(self.device)
        self.args.model_name = 'goal_pretrain'
        # Prepare log curve file and initialize best validation metrics
        self.log_curve_file = os.path.join(self.args.model_dir, 'pretrain_log_curve.txt')

        # Best metrics
        self.best_metrics = self.net.init_best_metrics()
        self.best_metrics_epochs = {k: -1 for k in self.best_metrics.keys()}

    def _set_optimizer(self, optimizer_name: str, parameters):
        """
        Set selected optimizer
        """
        if optimizer_name == 'Adam':
            return torch.optim.Adam(
                parameters, lr=self.args.learning_rate)
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(
                parameters, lr=self.args.learning_rate)
        elif optimizer_name == 'AdamW':
            return torch.optim.AdamW(
                parameters, lr=self.args.learning_rate)
        else:
            raise NameError(f'Optimizer {optimizer_name} not implemented.')

    def _set_scheduler(self, scheduler_name: str):
        """
        Set selected scheduler
        """
        # ReduceOnPlateau
        if scheduler_name == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=50,
                min_lr=1e-6,
                verbose=True)
        # Exponential
        elif scheduler_name == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.99)
        # CosineAnnealing
        elif scheduler_name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=300,
                eta_min=1e-5)
        else:  # Not set
            return None

    def _save_checkpoint(self, epoch, best_epoch=False):
        """
        Save model and optimizer states
        """
        saved_models_path = os.path.join(self.args.model_dir, 'saved_models')
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)
        # Save current checkpoint
        if not best_epoch:
            saved_model_name = os.path.join(
                saved_models_path,
                self.args.model_name + '_epoch_' +
                str(epoch).zfill(3) + '.pt')
        else:  # best model name
            saved_model_name = os.path.join(
                saved_models_path,
                self.args.model_name + '_best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
        }, saved_model_name)

    def _load_checkpoint(self, load_checkpoint):
        """
        Load a pre-trained model. Can then be used to test or resume training.
        """
        if load_checkpoint is not None:
            # Create load model path
            if load_checkpoint == 'best':
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models',
                    self.args.model_name + '_best_model.pt')
            else:  # Load specific checkpoint
                assert int(load_checkpoint) > 0, \
                    "Check args.load_model. Must be an integer > 0"
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models',
                    self.args.model_name + '_epoch_' +
                    str(load_checkpoint).zfill(3) + '.pt')
            print("\nSaved model path:", saved_model_name)
            # Load model
            if os.path.isfile(saved_model_name):
                print('Loading checkpoint ...')
                checkpoint = torch.load(saved_model_name,
                                        map_location=self.device)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(
                    checkpoint['model_state_dict'])
                print('Loaded checkpoint at epoch', model_epoch, '\n')
                return model_epoch
            else:
                raise ValueError("No such pre-trained model:", saved_model_name)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')

    def _load_or_restart(self):
        """
        Load a pre-trained model to resume training or restart from scratch.
        Can start from scratch or resume training, depending on input
        self.args.load_checkpoint parameter.
        """
        # load pre-trained model to resume training
        if self.args.load_checkpoint is not None:
            loaded_epoch = self._load_checkpoint(self.args.load_checkpoint)
            # start from the following epoch
            start_epoch = int(loaded_epoch) + 1
        else:
            start_epoch = 1
            # log_file header only the first time
            with open(self.log_curve_file, 'w') as f:
                f.write("epoch,learning_rate,valid_goal_BCE,valid_ADE_world_traj,valid_FDE_world_traj" +
                        ",".join(sorted(self.net.init_losses().keys())) +
                        "\n")
        return start_epoch

    def test(self, load_checkpoint):
        """
        Load a trained model and test it on the test set.
        """
        print('*** Test phase started ***')
        # some models do not need to be trained nor loaded
        if self.net.is_trainable:
            best_epoch = self._load_checkpoint(load_checkpoint)
        else:
            best_epoch = load_checkpoint
        print('Testing ...')
        total_results = []
        for run_idx in range(self.args.num_test_runs):
            print(f"\nTest run #{run_idx} ...")
            test_metrics = self._evaluate_epoch(best_epoch, mode='valid')
            run_results = dict(**test_metrics)
            # print losses and metrics for run i
            print(f'Test_set: {self.args.test_set},',
                  f'test_run_idx: {run_idx},',
                  f'epoch: {load_checkpoint},',
                  ', '.join([f"{k}={v:.5f}" for k, v in run_results.items()]))
            total_results.append(run_results)
        average_results = {k: np.mean([i[k] for i in total_results])
                           for k in run_results}
        # print average losses and metrics for
        print("\n" + "#"*25)
        print("#"*5 + " FINAL RESULTS " + "#"*5)
        print("#" * 25)
        print(f'Test_set: {self.args.test_set},',
              f'epoch: {load_checkpoint},',
              ', '.join([f"{k}={v:.5f}" for k, v in average_results.items()]))

    def train(self):
        """
        Train the model. Wrapper for train_loop.
        """
        # find where to start
        start_epoch = self._load_or_restart()

        # print model info
        print_model_summary(self.net)

        # parameters to update
        params = find_trainable_layers(self.net)

        # Set optimizer
        self.optimizer = self._set_optimizer(self.args.optimizer, params)
        # Set scheduler
        self.scheduler = self._set_scheduler(self.args.scheduler)

        # start training
        self._train_loop(start_epoch=start_epoch,
                         end_epoch=self.args.num_epochs)

    def train_test(self):
        """
        Perform training and then test on the best validation epoch.
        """
        self.train()
        print()
        self.test(load_checkpoint='best')

    def _train_loop(self, start_epoch, end_epoch):
        """
        Train the model. Loop over the epochs, train and update network
        parameters, save model, check results on validation set,
        print results and save log data.
        """
        # saved metrics before validation begins
        valid_metrics = {"valid_goal_BCE": 0, "valid_ADE_world_traj": 0, "valid_FDE_world_traj": 0}

        # initial learning rate
        if self.scheduler is not None:
            learning_rate = self.optimizer.param_groups[0]['lr']
        else:
            learning_rate = self.args.learning_rate

        phase_name = 'Train'
        best_metric_name = self.net.best_valid_metric()

        print(f'*** {phase_name} phase started ***')
        print(f"Starting epoch: {start_epoch}, final epoch: {end_epoch}")


        for epoch in range(start_epoch, end_epoch + 1):
            start_time = time.time()  # time epoch
            train_losses = self._train_epoch(epoch)

            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)  # save model
                print(f"Saved checkpoint at epoch {epoch}")

            # validation
            if epoch >= self.args.start_validation and \
                    epoch % self.args.validate_every == 0:
                valid_metrics = self._evaluate_epoch(epoch, mode='valid')

                # comment some of this print, if it is too long
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]) + ',',
                      ', '.join([f"{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 valid_metrics.items()]))

                # update best model and best metrics
                for k, v in self.best_metrics.items():
                    current_metric_loss = valid_metrics["valid_" + k]

                    if current_metric_loss < v:
                        self.best_metrics[k] = current_metric_loss
                        self.best_metrics_epochs[k] = epoch
                        # save best model on best metric
                        if k == best_metric_name:
                            self._save_checkpoint(epoch, best_epoch=True)
                            print(f"Saved best model at epoch {epoch}")

                print(', '.join([f"best_{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 self.best_metrics.items()]) + ',',
                      ', '.join([f"best_epoch_{metric_name}="
                                 f"{metric_epoch}" for
                                 metric_name, metric_epoch in
                                 self.best_metrics_epochs.items()]))
            else:
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]))

            # Update learning rate value
            if self.scheduler is not None:
                if self.args.scheduler == 'ReduceLROnPlateau':
                    if epoch >= self.args.start_validation and \
                            epoch % self.args.validate_every == 0:
                        lr_sched_metric = valid_metrics["valid_goal_BCE"]
                        self.scheduler.step(lr_sched_metric)
                elif self.args.scheduler in ['ExponentialLR',
                                             'CosineAnnealingLR']:
                    self.scheduler.step()
                learning_rate = self.optimizer.param_groups[0]['lr']

            # save metrics to log_curve.txt
            with open(self.log_curve_file, 'a') as f:
                f.write(','.join(str(m) for m in [
                    epoch, learning_rate,
                    valid_metrics["valid_goal_BCE"],
                    valid_metrics["valid_ADE_world_traj"],
                    valid_metrics["valid_FDE_world_traj"]] +
                    [train_losses[loss_name] for loss_name in sorted(
                        train_losses)]) + '\n')

    def _train_epoch(self, epoch):
        """
        Train one epoch of the model on the whole training set.
        """
        self.net.train()  # train mode

        # INIT LOSSES and METRICS
        losses_epoch = self.net.init_losses()
        losses_coeffs = self.net.set_losses_coeffs()
        # print('losses_coeffs: ', losses_coeffs)
        # Progress bar
        train_bar = tqdm(self.data_loaders['train'], ascii=True, ncols=100,
                         desc=f'Epoch {epoch}. Train batches')
        num_train_batches = len(self.data_loaders['train'])


        for batch_data, batch_id in train_bar:

            inputs, _, seq_list = \
                self.net.prepare_inputs(batch_data, batch_id)
            del batch_data

            self.optimizer.zero_grad()  # sets grads to zero

            losses = self.net.get_loss(inputs, seq_list) 
            loss = torch.zeros(1).to(self.device)
            for loss_name, loss_value in losses.items():
                loss += losses_coeffs[loss_name]*loss_value
                losses_epoch[loss_name] += loss_value.item()

            # Update network weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

            del inputs, batch_id
            del seq_list, losses, loss

        # update losses
        for loss_name in losses_epoch.keys():
            # mean losses over batches
            losses_epoch[loss_name] = losses_epoch[loss_name] / num_train_batches
        losses_epoch = add_dict_prefix(losses_epoch, prefix='train')

        return losses_epoch

    @torch.no_grad()
    def _evaluate_epoch(self, epoch, mode='valid'):
        """
        Loop over the validation or test set once. Compute metrics and save
        output trajectories.
        """
        self.net.eval()  # evaluation mode

        # INIT LOSSES and METRICS
        metrics_epoch = self.net.init_test_metrics()

        # Progress bar
        evaluate_bar = tqdm(self.data_loaders[mode], ascii=True, ncols=100,
                            desc=f'Epoch {epoch}. {mode.title()} batches')

        for batch_data, batch_id in evaluate_bar:

            inputs, ground_truth, seq_list = \
                self.net.prepare_inputs(batch_data, batch_id)
            del batch_data

            # compute metric_mask
            metric_mask = compute_metric_mask(seq_list)
            predictions = self.net.forward(
                inputs) # (21,Tp+Tf,B,2) 

            # update metrics
            for metric_name in metrics_epoch.keys():
                # print(metric_name)
                metrics_epoch[metric_name].extend(
                    self.net.compute_model_metrics(
                        metric_name=metric_name,
                        predictions=predictions,
                        ground_truth = ground_truth,
                        seq_list=seq_list,
                        metric_mask=metric_mask,
                        inputs=inputs,
                    ))
            del inputs, batch_id, predictions, ground_truth
            del seq_list, metric_mask

        # update metrics
        evaluate_metrics = {}
        for metric_name in metrics_epoch.keys():
            array_metric = np.array(metrics_epoch[metric_name])
            evaluate_metrics[metric_name] = array_metric.mean()
            if "goal" in metric_name:
                evaluate_metrics[metric_name + '_std'] = array_metric.std()
        evaluate_metrics = add_dict_prefix(evaluate_metrics, prefix=mode)
        return evaluate_metrics

    def _set_device(self):
        """
        Set the device for the experiment. GPU if available, else CPU.
        """
        # torch.cuda.is_available() already checked in parsed args
        device = torch.device(self.args.device)
        print('\nUsing device:', device)

        # Additional info when using cuda
        if device.type == 'cuda':
            print('Number of available GPUs:', torch.cuda.device_count())
            print('GPU name:', torch.cuda.get_device_name(0))
            print('Cuda version:', torch.version.cuda)
        print()
        return device
