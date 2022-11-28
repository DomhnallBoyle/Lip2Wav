from datetime import datetime

import numpy as np
import os
import pystoi
import tensorflow as tf
import time
import traceback
from jiwer import cer as calculate_cer
from tqdm import tqdm

from audio_utils import asr_transcribe
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from synthesizer.hparams import hparams_debug_string
from synthesizer.feeder import Feeder, _batches_per_group
from synthesizer.models import create_model
from synthesizer.utils import ValueWindow, plot
from synthesizer import infolog, audio

log = infolog.log


def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
    # Create tensorboard projector
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = checkpoint_path
    
    for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
        # Initialize config
        embedding = config.embeddings.add()
        # Specifiy the embedding variable and the metadata
        embedding.tensor_name = embedding_name
        embedding.metadata_path = path_to_meta
    
    # Project the embeddings to space dimensions for visualization
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)


def add_train_stats(model, hparams):
    with tf.variable_scope("stats") as scope:
        for i in range(hparams.tacotron_num_gpus):
            tf.summary.histogram("mel_outputs %d" % i, model.tower_mel_outputs[i])
            tf.summary.histogram("mel_targets %d" % i, model.tower_mel_targets[i])
        tf.summary.scalar("before_loss", model.before_loss)
        tf.summary.scalar("after_loss", model.after_loss)
        
        if hparams.predict_linear:
            tf.summary.scalar("linear_loss", model.linear_loss)
            for i in range(hparams.tacotron_num_gpus):
                tf.summary.histogram("mel_outputs %d" % i, model.tower_linear_outputs[i])
                tf.summary.histogram("mel_targets %d" % i, model.tower_linear_targets[i])
        
        tf.summary.scalar("regularization_loss", model.regularization_loss)
        #tf.summary.scalar("stop_token_loss", model.stop_token_loss)
        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("learning_rate", model.learning_rate)  # Control learning rate decay speed
        tf.summary.scalar("teacher_forcing_ratio", model.ratio)  # Control teacher forcing
        # ratio decay when mode = "scheduled"
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram("gradient_norm", gradient_norms)
        tf.summary.scalar("max_gradient_norm", tf.reduce_max(gradient_norms))  # visualize 
        # gradients (in case of explosion)

        # tf.summary.scalar('stoi', model.stoi)
        # tf.summary.scalar('estoi', model.estoi)

        if hparams.speaker_disentanglement:
            tf.summary.scalar('sdc_loss', model.sdc_loss)

        return tf.summary.merge_all()


def add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, loss, stoi, estoi, cer):
    values = [
        tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_before_loss", simple_value=before_loss),
        tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_after_loss", simple_value=after_loss),
        # tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/stop_token_loss", simple_value=stop_token_loss),
        tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_loss", simple_value=loss),
        tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_stoi', simple_value=stoi),
        tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_estoi', simple_value=estoi)
    ]
    if linear_loss is not None:
        values.append(tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_linear_loss", simple_value=linear_loss))
    if cer is not None:
        values.append(tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_cer", simple_value=cer))

    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def model_train_mode(args, feeder, hparams, global_step):
    with tf.variable_scope("Tacotron_model", reuse=tf.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.inputs, feeder.input_lengths, feeder.speaker_embeddings, 
                         feeder.mel_targets, targets_lengths=feeder.targets_lengths, global_step=global_step,
                         is_training=True, split_infos=feeder.split_infos, speaker_targets=feeder.speaker_targets)
        print("Model is initialized....")
        model.add_loss()
        print("Loss is added.....")
        model.add_optimizer(global_step)
        print("Optimizer is added....")
        stats = add_train_stats(model, hparams)

        return model, stats


def model_test_mode(args, feeder, hparams, global_step):
    #  I think the variable_scope allows sharing of variables i.e. weights across training and test models
    with tf.variable_scope("Tacotron_model", reuse=tf.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, 
                         feeder.eval_speaker_embeddings, feeder.eval_mel_targets, targets_lengths=feeder.eval_targets_lengths, 
                         global_step=global_step, is_training=False, is_evaluating=True,
                         split_infos=feeder.eval_split_infos)
        model.add_loss()

        return model


def train(log_dir, args, hparams):
    save_dir = os.path.join(log_dir, "taco_pretrained")
    plot_dir = os.path.join(log_dir, "plots")
    wav_dir = os.path.join(log_dir, "wavs")
    mel_dir = os.path.join(log_dir, "mel-spectrograms")
    eval_dir = os.path.join(log_dir, "eval-dir")
    eval_plot_dir = os.path.join(eval_dir, "plots")
    eval_wav_dir = os.path.join(eval_dir, "wavs")
    tensorboard_dir = os.path.join(log_dir, "tacotron_events")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    checkpoint_fpath = os.path.join(save_dir, "tacotron_model.ckpt")

    # init the log file
    infolog.init(os.path.join(log_dir, f'train.log.{args.log_number}'), args.name)

    log("Checkpoint path: {}".format(checkpoint_fpath))
    log("Using model: Tacotron")
    log(hparams_debug_string())
    log(args.__dict__)

    # Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = Feeder(
            coord,
            hparams,
            num_test_batches=args.num_test_batches,
            apply_augmentation=args.apply_augmentation,
            training_sample_pool_location=args.training_sample_pool_location,
            val_sample_pool_location=args.val_sample_pool_location,
            use_selection_weights=args.use_selection_weights
        )
    
    # Set up model:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    model, stats = model_train_mode(args, feeder, hparams, global_step)
    eval_model = model_test_mode(args, feeder, hparams, global_step)
    
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=2)

    log("Tacotron training set to a maximum of {} steps".format(args.tacotron_train_steps))
    
    # Memory allocation on the GPU as needed
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)  # ~80% of total GPU memory
    # config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    # Train
    with tf.Session(config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            
            sess.run(tf.global_variables_initializer())
            
            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    if args.checkpoint_path:
                        log(f'Loading checkpoint {args.checkpoint_path}')
                        saver.restore(sess, args.checkpoint_path)
                    else:
                        checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                        if checkpoint_state and checkpoint_state.model_checkpoint_path:
                            log("Loading checkpoint {}".format(checkpoint_state.model_checkpoint_path),
                                slack=True)
                            saver.restore(sess, checkpoint_state.model_checkpoint_path)
                            # log("Loading checkpoint {}".format(hparams.eval_ckpt), slack=True)
                            # saver.restore(sess, hparams.eval_ckpt)
                        else:
                            log("No model to load at {}".format(save_dir), slack=True)
                            saver.save(sess, checkpoint_fpath, global_step=global_step)
                
                except tf.errors.OutOfRangeError as e:
                    log("Cannot restore checkpoint: {}".format(e), slack=True)
            else:
                log("Starting new training!", slack=True)
                saver.save(sess, checkpoint_fpath, global_step=global_step)
            
            # initializing feeder
            feeder.start_threads(sess)
            print("Feeder is initialized....")
            print("Ready to train....")

            # Training loop
            while not coord.should_stop() and step < args.tacotron_train_steps:

                feeder.dequeue_training_sample()

                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)  # loss appended after every step (keeps window of 100 losses)
                message = "Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]".format(
                    step, time_window.average, loss, loss_window.average)
                log(message, end="\r")

                if loss > 100 or np.isnan(loss):
                    log("Loss exploded to {:.5f} at step {}".format(loss, step))
                    raise Exception("Loss exploded")

                if step == 100 or step % args.summary_interval == 0:
                    log("\nWriting summary at step {}".format(step))
                    feeder.dequeue_training_sample()
                    summary_writer.add_summary(sess.run(stats), step)

                if step == 100 or step % args.eval_interval == 0:
                    # Run eval and save eval stats
                    log("\nRunning evaluation at step {}".format(step))

                    eval_losses = []
                    before_losses = []
                    after_losses = []
                    linear_losses = []
                    linear_loss = None
                    eval_stois = []
                    eval_estois = []
                    eval_cers = []

                    use_cer_metric = args.use_cer_metric and (step == 100 or step % args.cer_interval == 0)

                    if hparams.predict_linear:
                        for i in tqdm(range(feeder.test_steps)):
                            eloss, before_loss, after_loss, stop_token_loss, linear_loss, mel_p, \
							mel_t, t_len, align, lin_p, lin_t = sess.run([
                                    eval_model.tower_loss[0], eval_model.tower_before_loss[0],
                                    eval_model.tower_after_loss[0],
                                    eval_model.tower_stop_token_loss[0],
                                    eval_model.tower_linear_loss[0],
                                    eval_model.tower_mel_outputs[0][0],
                                    eval_model.tower_mel_targets[0][0],
                                    eval_model.tower_targets_lengths[0][0],
                                    eval_model.tower_alignments[0][0],
                                    eval_model.tower_linear_outputs[0][0],
                                    eval_model.tower_linear_targets[0][0],
                                ])
                            eval_losses.append(eloss)
                            before_losses.append(before_loss)
                            after_losses.append(after_loss)
                            stop_token_losses.append(stop_token_loss)
                            linear_losses.append(linear_loss)
                        linear_loss = sum(linear_losses) / len(linear_losses)

                        wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
                        audio.save_wav(wav, os.path.join(eval_wav_dir,
                                                         "step-{}-eval-wave-from-linear.wav".format(
                                                             step)), sr=hparams.sample_rate)

                    else:
                        feeder.set_test_feeding_status(True)
                        # TODO: Race condition here
                        #  worker: if i = 15 (< 16), batch loaded
                        #  main: unload + calc stois, goes back into loop (True), wait for next batch to be loaded
                        #  worker: i = 16 (== 16), breaks loading while, never loads next batch
                        #  stuck main thread waiting for batch
                        while feeder.get_test_feeding_status():
                            feeder.set_load_next_test_sample(True)  # load next batch
                            # this executes a batch at a time
                            # would need to loop over this for each batch
                            eloss, before_loss, after_loss, mel_ps, mel_ts, t_lens, aligns = sess.run([
                                eval_model.tower_loss[0],
                                eval_model.tower_before_loss[0],
                                eval_model.tower_after_loss[0],
                                eval_model.tower_mel_outputs[0],
                                eval_model.tower_mel_targets[0],
                                eval_model.tower_targets_lengths[0],
                                eval_model.tower_alignments[0]
                            ])
                            eval_losses.append(eloss)
                            before_losses.append(before_loss)
                            after_losses.append(after_loss)

                            # calculate average STOI between groundtruth and generated batch mel-specs
                            # calculate CER using DeepSpeech ASR
                            # done for a SINGLE batch
                            stoi, estoi, cer = 0, 0, 0
                            for gt_melspec, gen_melspec in tqdm(zip(mel_ts, mel_ps)):
                                gt = audio.inv_mel_spectrogram(gt_melspec.T, hparams)
                                gen = audio.inv_mel_spectrogram(gen_melspec.T, hparams)

                                if len(gt) > len(gen):
                                    gt = gt[:gen.shape[0]]
                                elif len(gen) > len(gt):
                                    gen = gen[:gt.shape[0]]

                                stoi += pystoi.stoi(gt, gen, hparams.sample_rate, extended=False)
                                estoi += pystoi.stoi(gt, gen, hparams.sample_rate, extended=True)

                                if use_cer_metric:
                                    audio.save_wav(gt, '/tmp/gt.wav', hparams.sample_rate)
                                    audio.save_wav(gen, '/tmp/gen.wav', hparams.sample_rate)
                                    gt_prediction = asr_transcribe('/tmp/gt.wav')[0]
                                    gen_prediction = asr_transcribe('/tmp/gen.wav')[0]
                                    try:
                                        cer += calculate_cer(gt_prediction, gen_prediction)
                                    except ValueError:
                                        log(f'CER failed: {gt_prediction}, {gen_prediction}')

                            stoi /= len(mel_ts)
                            estoi /= len(mel_ts)
                            cer /= len(mel_ts)

                            eval_stois.append(stoi)
                            eval_estois.append(estoi)
                            eval_cers.append(cer)

                    eval_loss = sum(eval_losses) / len(eval_losses)
                    before_loss = sum(before_losses) / len(before_losses)
                    after_loss = sum(after_losses) / len(after_losses)
                    eval_stoi = sum(eval_stois) / len(eval_stois)
                    eval_estoi = sum(eval_estois) / len(eval_estois)
                    eval_cer = sum(eval_cers) / len(eval_cers) if use_cer_metric else None

                    # print(mel_ps.shape)  # (32, 80, 80)

                    # just take the first example from a test batch (could be different because of shuffling)
                    mel_p = mel_ps[0]
                    mel_t = mel_ts[0]
                    t_len = t_lens[0]
                    align = aligns[0]

                    if step % args.checkpoint_interval == 0:
                        log("Saving eval log to {}..".format(eval_dir))
                        # Save some log to monitor model improvement on same unseen sequence
                        wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
                        audio.save_wav(wav, os.path.join(eval_wav_dir,
                                                         "step-{}-eval-wave-from-mel.wav".format(step)),
                                       sr=hparams.sample_rate)

                        plot.plot_alignment(align, os.path.join(eval_plot_dir,
                                                                "step-{}-eval-align.png".format(step)),
                                            title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                        time_string(),
                                                                                        step,
                                                                                        eval_loss),
                                            max_len=t_len // hparams.outputs_per_step)
                        plot.plot_spectrogram(mel_p, os.path.join(eval_plot_dir,
                                                                  "step-{}-eval-mel-spectrogram.png".format(step)),
                                              title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                          time_string(),
                                                                                          step,
                                                                                          eval_loss),
                                              target_spectrogram=mel_t,
                                              max_len=t_len)

                        if hparams.predict_linear:
                            plot.plot_spectrogram(lin_p, os.path.join(eval_plot_dir,
                                                                      "step-{}-eval-linear-spectrogram.png".format(
                                                                          step)),
                                                  title="{}, {}, step={}, loss={:.5f}".format(
                                                      "Tacotron", time_string(), step, eval_loss),
                                                  target_spectrogram=lin_t,
                                                  max_len=t_len, auto_aspect=True)

                    log("Eval loss for global step {}: {:.3f}".format(step, eval_loss))
                    log("Writing eval summary!")
                    add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, eval_loss, eval_stoi,
                                   eval_estoi, eval_cer)

                if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or step == 300:

                    feeder.dequeue_training_sample()

                    # Save model and current global step
                    saver.save(sess, checkpoint_fpath, global_step=global_step)
                    
                    log("\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..")
                    input_seq, mel_prediction, alignment, target, target_length = sess.run([
                        model.tower_inputs[0][0],
                        model.tower_mel_outputs[0][0],
                        model.tower_alignments[0][0],
                        model.tower_mel_targets[0][0],
                        model.tower_targets_lengths[0][0],
                    ])
                    
                    # save predicted mel spectrogram to disk (debug)
                    mel_filename = "mel-prediction-step-{}.npy".format(step)
                    np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T,
                            allow_pickle=False)
                    
                    # save griffin lim inverted wav for debug (mel -> wav)
                    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
                    audio.save_wav(wav,
                                   os.path.join(wav_dir, "step-{}-wave-from-mel.wav".format(step)),
                                   sr=hparams.sample_rate)
                    
                    # save alignment plot to disk (control purposes)
                    plot.plot_alignment(alignment,
                                        os.path.join(plot_dir, "step-{}-align.png".format(step)),
                                        title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                    time_string(),
                                                                                    step, loss),
                                        max_len=target_length // hparams.outputs_per_step)
                    # save real and predicted mel-spectrogram plot to disk (control purposes)
                    plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir,
                                                                       "step-{}-mel-spectrogram.png".format(
                                                                           step)),
                                          title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                      time_string(),
                                                                                      step, loss),
                                          target_spectrogram=target,
                                          max_len=target_length)
                    # log("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
                
                if step % args.embedding_interval == 0 or step == args.tacotron_train_steps or step == 1:
                    # Get current checkpoint state
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    
                    # Update Projector
                    #log("\nSaving Model Character Embeddings visualization..")
                    #add_embedding_stats(summary_writer, [model.embedding_table.name],
                    #                   [char_embedding_meta],
                    #                    checkpoint_state.model_checkpoint_path)
                    #log("Tacotron Character embeddings have been updated on tensorboard!")
            
            log("Tacotron training complete after {} global steps!".format(
                args.tacotron_train_steps), slack=True)
            return save_dir
        
        except Exception as e:
            log("Exiting due to exception: {}".format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def tacotron_train(args, log_dir, hparams):
    return train(log_dir, args, hparams)
