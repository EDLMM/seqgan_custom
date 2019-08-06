#-*- coding:utf-8 -*-
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
import codecs
import sys

IS_RESUME=False

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 50 #32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH =82 #32 #82 # sequence length, need to set up according to data set before run
START_TOKEN = 1
PRE_EPOCH_NUM = 120 #120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64 # 2

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 32, 64] #[1,2,3,4,5,6] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] #最后filter不能比 sequence 长
dis_num_filters =  [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160,160,200]# [100, 200, 200, 200, 200, 100]#[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160] 这个和上面的对应
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64
DIS_PRE_EPOCH_NUM=50 #50
#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200 #200
positive_file = './save/imdb_pos_tokens.txt' # './save/overfit_test.txt'#'save/tokenized_data.txt'#'save/real_data.txt'
negative_file = './save/generator_sample.txt'
eval_file = './save/eval_file.txt'
chkpt_dir= "./checkpoint/"
generated_num = 500 # for snli 10000 #2 for overfit #42068 for ptb # should be the same size as the true data


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

def mplt_single(x_axis,y_axis,save_name="some_plotting"):
    # fig, axs = plt.subplots(1, 1, constrained_layout=True,figsize=(5,6))
    # fig.suptitle(save_name)
    plt.plot(x_axis,y_axis,'r-')
    # axs[0].set_title('Generator '+phase_name+ ' loss')
    plt.ylabel(save_name)
    plt.xlabel('epoch')

    plt.savefig(save_name+"_plotting.png")
    plt.close()

# load pre-trained embeddings
def load_embd(filepath_glove = './save/glove.6B.50d.txt'):
    #Load GLOVE vectors
    glove_vocab = []
    glove_embd=[]
    embedding_dict = {}
    file = codecs.open(filepath_glove,'r')#,encoding='UTF-8'
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]] # convert to list of float
        embedding_dict[vocab_word]=embed_vector
    file.close()
    print 'Loaded GLOVE'
    # glove_vocab_size = len(glove_vocab)
    # embedding_dim = len(embed_vector)
    return glove_vocab, embedding_dict

def get_adapted_embed(embed_vocab,embedding_dict,train_vocab):
    embeddings_tmp=[]

    for key,value in train_vocab.items():
        if key in embed_vocab:
            embeddings_tmp.append(embedding_dict[key])
        else:
            # if it's not in the pre-trained embeddings, initialize with random floats
            rand_num = np.random.uniform(low=-0.2, high=0.2,size=EMB_DIM)
            embeddings_tmp.append(rand_num)

    embedding = np.asarray(embeddings_tmp)

    return embedding

def main():
    
    print '#########################################################################'
    print "version for IMDB postivie sentences with plot accuracy 64"
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

    # get pre-trained embeddings for training data
    glove_vocab, embedding_dict=load_embd()
    train_embeddings=get_adapted_embed(glove_vocab,embedding_dict,vocab_dict)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,randomEmb=False)
    # TODO: 在generator内部加 tree和embedding2index；测试是否真的能把embedding load 到模型里面；discriminator也要load吗?应该要
    # target_params = cPickle.load(open('save/target_params.pkl'))
    # target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    # load the pre-trained embeddings into model
    sess.run(generator.embedding_init,feed_dict={generator.embedding_placeholder: train_embeddings})

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate_samples(sess, target_lstm, BATCH_SIZE, 2, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    p_G_MLE_loss=[]
    p_G_MLE_epoch=[]
    p_D_MLE_loss=[]
    p_D_MLE_epoch=[]
    p_D_MLE_accuracy=[]
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
            save_path = saver.save(sess, chkpt_dir+"model.ckpt")
            print("Model saved in path: %s" % save_path)

        p_G_MLE_loss.append(loss)
        p_G_MLE_epoch.append(epoch)
          # Save the variables to disk.
        
    
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
        daccuracy=[]
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _,eloss,eacrcy = sess.run([discriminator.train_op,discriminator.loss,discriminator.accuracy], feed)
                dloss.append(eloss)
                daccuracy.append(eacrcy)
        daccuracy=np.mean(daccuracy)
        dloss=np.mean(dloss)
        p_D_MLE_loss.append(dloss)
        p_D_MLE_accuracy.append(daccuracy)
        p_D_MLE_epoch.append(epoch)
        if epoch % 10 ==0:
            print "pre-train discriminator {}, loss: {}".format(epoch,dloss)
            log.write("pre-train discriminator {}, loss: {}".format(epoch,dloss))
            save_path = saver.save(sess, chkpt_dir+"model.ckpt")
            print("Model saved in path: %s" % save_path)


    rollout = ROLLOUT(generator, 0.8)
    mplt_single(p_D_MLE_epoch,p_D_MLE_accuracy,"MLE pre-train discriminator accuracy")
    mplt_save(p_G_MLE_epoch,p_G_MLE_loss,p_D_MLE_epoch,p_D_MLE_loss,"MLE pre-train on SNLI_test")
    
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

    p_D_AD_Acurracy=[]
    
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
        p_G_AD_epoch.append(total_batch)
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
        dloss=[]
        daccuracy=[]
        for depoch in range(D_AD_EPOCH_NUM):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _,eloss,eacrcy = sess.run([discriminator.train_op,discriminator.loss,discriminator.accuracy], feed)
                    dloss.append(eloss)
                    daccuracy.append(eacrcy)

        ddloss=np.mean(dloss)
        ddaccuracy=np.mean(daccuracy)
        p_D_AD_loss.append(ddloss)
        p_D_AD_Acurracy.append(ddaccuracy)
        p_D_AD_epoch.append(total_batch)

        if total_batch % 10 == 0 or total_batch==TOTAL_BATCH-1:
            mplt_save(p_G_AD_epoch,p_G_AD_loss,p_D_AD_epoch,p_D_AD_loss,"Adversarial Training on SNLI_test")
            mplt_single(p_D_AD_epoch,p_D_AD_Acurracy,"Discriminator_Acurracy");
            save_path = saver.save(sess, chkpt_dir+"model.ckpt")
            print("Model saved in path: %s" % save_path)
            samples=index2sent(generator.generate(sess),vocab_res)
            print "generate samples at NO.{} epoch:".format(total_batch)
            pp.pprint(samples)
            log.write( "generate samples at NO.{} epoch:".format(total_batch))
            samples_to_log(log,samples)
    log.close()

def gen_test():
    print '#########################################################################'
    print 'Restore from trained model and generate sentences'
    print '#########################################################################'

    vocab_dict, vocab_res = data_utils_en.load_vocab('./save/vocab.txt')
    vocab_size = len(vocab_dict)
    # tf.reset_default_graph()

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,randomEmb=False)
    # discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                # filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,chkpt_dir+"model.ckpt")
        print("Model restored.")
        samples=index2sent(generator.generate(sess),vocab_res)
        print "generate samples from restored model:"
        pp.pprint(samples)



if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == "gen":
        gen_test()
    else:
        main()
    
