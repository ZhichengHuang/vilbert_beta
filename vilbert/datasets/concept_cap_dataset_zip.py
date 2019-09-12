import zipfile 
import os
from pandas import Series
import io
import random
import copy

import torch
from torch.utils.data import Dataset 
import numpy as np 
import pickle
import contextlib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


def get_dist_info():
    initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def check_file_exist(filename):
    if not os.path.isfile(filename):
        # raise IOError(msg_tmpl.format(filename))
        return False
    return True

class TxtReader(object):
    def __init__(self):
        super(TxtReader, self).__init__()
        self.id_context = Series()

    def read(self, txt_file, position, pid):
        key_name = txt_file + "_" + str(pid)

        if key_name in self.id_context:
            self.id_context[key_name].seek(position, os.SEEK_SET)
            return self.id_context[key_name].readline()
        else:
            file_handle = open(txt_file, 'r',encoding='utf-8')
            self.id_context[key_name] = file_handle
            self.id_context[key_name].seek(position, os.SEEK_SET)
            return self.id_context[key_name].readline()


class ZipReader(object):
    def __init__(self):
        super(ZipReader, self).__init__()
        self.id_context = Series()

    def read(self, zip_file, image_name, pid):
        key_name = zip_file + "_" + str(pid)

        if key_name in self.id_context:
            with self.id_context[key_name].open(image_name) as f:
                tmp = f.read()
            return tmp
        else:
            file_handle = zipfile.ZipFile(zip_file, 'r', zipfile.ZIP_LZMA)
            self.id_context[key_name] = file_handle
            return self.id_context[key_name].read(image_name)


class InputExample(object):
    """ A single training example for the lnaguage model"""
    def __init__(self,image_feat=None,image_target=None,caption=None,is_next=None,lm_labels=None,image_loc=None,num_boxes=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_next=None,
        lm_label_ids=None,
        image_feat=None,
        image_target=None,
        image_loc=None,
        image_label=None,
        image_mask=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask




class ConceptCapDataset(Dataset):
    def __init__(self,data_path,
                tokenizer,
                seq_len,
                region_len,
                data_split,
                predict_feature=False,
                visualization=False):
        """
        Load the dataset and sample the 
        """
        self.data_path = data_path
        self.data_split = data_split,
        self.region_len=region_len
        self.visualization = visualization
        self.seq_len=seq_len
        self.predict_feature=predict_feature
        
        self.zip_reader = ZipReader()
        self.language_reader = TxtReader()
        self.create_language_id2val()

        self.tokenizer = tokenizer



    def create_language_id2val(self):
        self.language_ids =[]
        if self.data_split == "train":
            if check_file_exist(os.path.join(self.data_path,"train_language_all_ids.pkl")):
                self.language_ids = pickle.load(open(os.path.join(self.data_path,"train_language_all_ids.pkl"),'rb'))
            else:
                for file in os.listdir(self.data_path):
                    if "train_" in file and file.endswith(".txt"):
                        data_full_path = os.path.join(self.data_path,file)
                        with open(data_full_path,'r', encoding='utf-8') as f:
                            file_pos = f.tell()
                            self.language_ids.extend([file+"#"+str(file_pos)])
                            while f.readline() !="":
                                file_pos = f.tell()
                                self.language_ids.extend([file+"#"+str(file_pos)])
                pickle.dump(self.language_ids,open(os.path.join(self.data_path,"train_language_all_ids.pkl"),'wb'))
        
        elif self.data_split == "val":
            if check_file_exist(os.path.join(self.data_path,"val_language_all_ids.pkl")):
                self.language_ids = pickle.load(open(os.path.join(self.data_path,"val_language_all_ids.pkl"),'rb'))
            else:
                for file in os.listdir(self.data_path):
                    if "val_" in file and file.endswith(".txt"):
                        data_full_path = os.path.join(self.data_path,file)
                        with open(data_full_path,'r', encoding='utf-8') as f:
                            file_pos = f.tell()
                            self.language_ids.extend([file+"#"+str(file_pos)])
                            while f.readline() !="":
                                file_pos = f.tell()
                                self.language_ids.extend([file+"#"+str(file_pos)])
                pickle.dump(self.language_ids,open(os.path.join(self.data_path,"val_language_all_ids.pkl"),'wb'))

    

    def random_cap(self, caption):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 0

        if random.random() > 0.5:
            label = 0
        else:
            caption = self.get_random_caption()
            label = 1

        return caption, label

    def get_random_caption(self):
        """
        Get random caption from annother documnet for nextSentence task
        :return: str, content for one line
        """

        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large corpora.
        # However, just to be careful, we try to make sure sure that the random document is not the same as the documnet we're processing
        # rand_doc_idx = random.randint(0,self.num_caps-1)
        # caption = self.captions[rand_doc_idx]
        rand_doc_idx = random.randint(0,len(self.language_ids)-1)

        language_id = self.language_ids[rand_doc_idx] 
        txt_file,position = language_id.split("#")
        tmp_line = self.language_reader.read(os.path.join(self.data_path,txt_file),int(position),0)

        ids,cap,features_name = self.process_line(tmp_line)

        return cap

    
    def convert_example_to_features(self,example,max_seq_length,tokenizer,max_region_length):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        self._truncate_seq_pair(caption, max_seq_length - 2)
        caption, caption_label = self.random_word(caption, tokenizer)

        image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
        lm_label_ids = [-1] + caption_label + [-1]
        # image_label = ([-1] + image_label)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        # for i in range(36):
        #     # tokens.append(0)
        #     segment_ids.append(0)

        # tokens.append("[SEP]")
        # segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        # if example.guid < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("LM label: %s " % (lm_label_ids))
        #     logger.info("Is next sentence label: %s " % (example.is_next))

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask = np.array(image_mask)
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()
    
    def random_word(self, tokens, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    logger.warning(
                        "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                    )
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label
    
    def random_region(self, image_feat, image_loc, num_boxes):
        """
        """
        output_label = []

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label

    def process_item(self, caption,img):
        image_feature,image_target,image_location_ws,image_h,image_w= img['feature'],img['label'],img['box'],img['width'],img['height']# w,h,w,h
        image_location = np.zeros((self.region_len,5),dtype=np.float32)
        image_location[:self.region_len,:4] = image_location_ws

        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))

        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        
        if self.predict_feature:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)            

        caption, label = self.random_cap(caption)

        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            image_loc=image_location,
            num_boxes=self.region_len
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)
        
        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
        )
        return cur_tensors
    

    def process_line(self,line):
        ids,caption,image=line.strip().split("\t")
        id=ids.split("id:")[-1]
        cap = caption.split("caption:")[-1]
        feature = image.split("Image:")[-1]
        return id,cap,feature


    
    def __getitem__(self,index):
        language_id = self.language_ids[index] 
        txt_file,position = language_id.split("#")
        tmp_line = self.language_reader.read(os.path.join(self.data_path,txt_file),int(position),0)
        ids,cap,features_name = self.process_line(tmp_line)
        feature_name = self.data_split+"_features/"+features_name.split(".jpg")[0]+".npy"
        tmp_image =self.zip_reader.read(os.path.join(self.data_path,self.data_split+"_features.zip"),feature_name,0)

        with contextlib.closing(io.BytesIO(tmp_image)) as f:
            img = np.load(f,all_pickle=True).item()
        
        out=self.process_item(cap,img)
        return out,ids

    def __len__(self):
        return len(self.language_ids)


def build_dataloader(data_path,
                tokenizer,
                seq_len,
                region_len,
                data_split,
                imgs_per_gpu=64,
                workers_per_gpu=4,
                shuffle=False,
                distributed=True,
                predict_feature=False,
                visualization=False):
    dataset = ConceptCapDataset(data_path,
                tokenizer,
                seq_len,
                region_len,
                data_split,
                predict_feature,
                visualization)
    if distributed:
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(dataset)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = Sampler(dataset)
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        #collate_fn=trim_collate,  # partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=True,
        **kwargs)
    return data_loader
    
    
        