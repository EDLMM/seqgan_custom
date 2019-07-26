from __future__ import division
import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
# from target_lstm import TARGET_LSTM
import cPickle
import pprint as pp
import data_utils_en
import matplotlib.pyplot as plt

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 82 # sequence length, need to set up according to data set before run
START_TOKEN = 1
PRE_EPOCH_NUM = 120 #120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 1 #64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64
DIS_PRE_EPOCH_NUM=50 #50
#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200 #200
positive_file =  './save/overfit_test.txt'#'save/tokenized_data.txt'#'save/real_data.txt'
negative_file = './save/generator_sample.txt'
eval_file = './save/eval_file.txt'
generated_num = 2 #42068 # should be the same size as the true data


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def samples_to_log(f,samples):
    for sample in samples:
        buffer = str(sample) + '\n'
        f.write(buffer)

def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
# my visualization add-ons
def index2sent(number_sents,vocab_res):
    sents=[]
    for number_sent in number_sents:
        sent=""
        for index in number_sent:
            if index != data_utils_en.EOS_ID and index != data_utils_en.GO_ID and index != data_utils_en.PAD_ID:
                        sent += vocab_res[index]
                        sent += str(' ')
            if index == data_utils_en.EOS_ID:
                # poem += str('\n')
                break
        # print sent
        sents.append(sent)
    return sents

def mplt_save(epoch1,loss1,epoch2,loss2,phase_name="MLE pretrain"):
    fig, axs = plt.subplots(2, 1, constrained_layout=True,figsize=(5,10))
    fig.suptitle(phase_name+' phase loss')
    
    axs[0].plot(epoch1,loss1,'g-')
    axs[0].set_title('Generator '+phase_name+ ' loss')
    axs[0].set_ylabel(phase_name+ ' loss')
    axs[0].set_xlabel('epoch')
    
    axs[1].plot(epoch2,loss2, '--')
    axs[1].set_xlabel('epoch')
    axs[1].set_title('Discriminator' +phase_name+ ' loss')
    axs[1].set_ylabel(phase_name+ ' loss')
    
    plt.savefig(phase_name+" loss.png")
    plt.close()

def main():
    
    print '#########################################################################'
    print "version for overfit with delayed LOSS plot"
    print '#########################################################################'

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 1

    vocab_dict, vocab_res = data_utils_en.load_vocab('./save/vocab.txt')
    # data = data_utils_en.load_data('./save/tokenized_data.txt')
    vocab_size = len(vocab_dict)

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    # likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    # target_params = cPickle.load(open('save/target_params.pkl'))
    # target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate_samples(sess, target_lstm, BATCH_SIZE, 2, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    p_G_MLE_loss=[]
    p_G_MLE_epoch=[]
    p_D_MLE_loss=[]
    p_D_MLE_epoch=[]
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        # if epoch % 5 == 0:
        #     generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
        #     likelihood_data_loader.create_batches(eval_file)
        #     test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        #     print 'pre-train epoch ', epoch, 'test_loss ', test_loss
        #     buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
        #     log.write(buffer)
        if epoch % 10 ==0:
            print "pre-train generator {}, loss: {}".format(epoch,loss)
            log.write("pre-train generator {}, loss: {}".format(epoch,loss))
        p_G_MLE_loss.append(loss)
        p_G_MLE_epoch.append(epoch)
    
    samples=index2sent(generator.generate(sess),vocab_res)
    print "generate samples after pretrain G:"
    pp.pprint(samples)
    log.write("\ngenerate samples after pretrain G:\n")
    samples_to_log(log,samples)
    
    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    for epoch in range(DIS_PRE_EPOCH_NUM):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        dloss=[]
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _,eloss = sess.run([discriminator.train_op,discriminator.loss], feed)
                dloss.append(eloss)
        dloss=np.mean(dloss)
        p_D_MLE_loss.append(dloss)
        p_D_MLE_epoch.append(epoch)
        if epoch % 10 ==0:
            print "pre-train discriminator {}, loss: {}".format(epoch,dloss)
            log.write("pre-train discriminator {}, loss: {}".format(epoch,dloss))


    rollout = ROLLOUT(generator, 0.8)
    
    mplt_save(p_G_MLE_epoch,p_G_MLE_loss,p_D_MLE_epoch,p_D_MLE_loss,"MLE pre-train")
    
    samples=index2sent(generator.generate(sess),vocab_res)
    print "generate samples after pretrain D:"
    pp.pprint(samples)
    log.write("generate samples after pretrain D:\n")
    samples_to_log(log,samples)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    p_G_AD_loss=[]
    p_G_AD_epoch=[]
    p_D_AD_loss=[]
    p_D_AD_epoch=[]
    
    G_AD_EPOCH_NUM=1
    D_AD_EPOCH_NUM=5
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        gloss=[]
        for it in range(G_AD_EPOCH_NUM):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _,egloss = sess.run([generator.g_updates, generator.g_loss],feed_dict=feed)
            gloss.append(egloss)
        gloss=np.mean(gloss)
        p_G_AD_loss.append(gloss)
        p_G_AD_epoch.append(total_batch*G_AD_EPOCH_NUM+it)
        # Test
        # if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
        #     generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
        #     likelihood_data_loader.create_batches(eval_file)
        #     test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        #     buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
        #     print 'total_batch: ', total_batch, 'test_loss: ', test_loss
        #     log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for depoch in range(D_AD_EPOCH_NUM):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            dloss=[]
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _,eloss = sess.run([discriminator.train_op,discriminator.loss], feed)
                    dloss.append(eloss)
            dloss=np.mean(dloss)
            p_D_AD_loss.append(dloss)
            p_D_AD_epoch.append(total_batch*D_AD_EPOCH_NUM + depoch)
            
        samples=index2sent(generator.generate(sess),vocab_res)
        print "generate samples at NO.{} epoch:".format(total_batch)
        pp.pprint(samples)
        log.write( "generate samples at NO.{} epoch:".format(total_batch))
        samples_to_log(log,samples)
        if total_batch % 20 == 0 or total_batch==TOTAL_BATCH-1:
            mplt_save(p_G_AD_epoch,p_G_AD_loss,p_D_AD_epoch,p_D_AD_loss,"Adversarial Training")
    
    log.close()


if __name__ == '__main__':
    main()
