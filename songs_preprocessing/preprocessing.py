from music21 import *
import matplotlib.pyplot as plt
import numpy as np
import glob

class Dataset():
    def __init__(self,path):
        "Path: path to the folder where midi files are stored"
        self.data_to_int = dict()
        self.int_to_data = dict()
        self.INPUT = 100
        self.HIDDEN = 256
        self.vocab_size = len(self.data_to_int)
        self.OUTPUT = self.vocab_size
        self.LEARNING_RATE = 0.005
        self.BETA1 = 0.9
        self.BETA2 = 0.999
        self.BATCH_SIZE = 1
        self.songs = glob.glob(path+"/*.mid")
        self.processed_data = None
        self.transformed_data = None
        self.data_to_int = dict()
        self.int_to_data = dict()
        self.train_data = None 

    def plot_song(self,path):
        """plots a single song
        
           Path: filepath to midi file.
        """
        midi_data = converter.parse('/content/sample_data/202.mid')
        midi_data.write('text')
        midi_data.plot()
        midi_data.plot('histogram', 'pitchclass')
    
    def process_data(self):
        whole_data = []
        for song in self.songs:
            midi_data = converter.parse(song).flat
            song_data = []
            prev_offset = -1
            for element in midi_data:
                if isinstance(element, note.Note):
                    if element.offset != prev_offset:
                        song_data.append([element.pitch.nameWithOctave, 
                                        element.quarterLength])
                    else:
                        if len(song_data[-1]) < 4:
                            song_data[-1].append(element.pitch.nameWithOctave)   
                            song_data[-1].append(element.quarterLength)       
                    prev_offset = element.offset
                elif isinstance(element, chord.Chord):
                    pitch_names = '.'.join(n.nameWithOctave for n in element.pitches)
                    if element.offset != prev_offset:
                        song_data.append([pitch_names, element.quarterLength])
                    else:
                        if len(song_data[-1]) < 4:
                            song_data[-1].append(pitch_names)   
                            song_data[-1].append(element.quarterLength)      
                    prev_offset = element.offset
            for item in song_data:
                if len(item) < 4:
                    item.append(None)
                    item.append(None)
            whole_data.append(song_data)
            self.processed_data=whole_data
        return whole_data

    def transform_data(self):
        max_len = 0
        for song in self.songs:
            max_len = max(max_len, len(song))
        for song in self.songs:
            for i in range(max_len - len(song)):
                song.append([None, None, None, None])
        transform_data = []
        for song in self.songs:
            t_song_data = []
            for item in song:
                t_song_data.append(tuple(item))
            transform_data.append(t_song_data)
        self.transformed_data = transform_data
        return transform_data
    def get_dictionary(self):
        possible_combs = set(item for song in self.songs for item in song)
        data_to_int = dict((v, i) for i, v in enumerate(possible_combs))
        int_to_data = dict((i, v) for i, v in enumerate(possible_combs))
        self.data_to_int = data_to_int
        self.int_to_data = int_to_data

        return data_to_int, int_to_data
    def get_batches(self):
        train_dataset = []
        for i in range(len(self.songs) - self.BATCH_SIZE + 1):
            start = i * self.BATCH_SIZE
            end = start + self.BATCH_SIZE
            batch_data = self.songs[start:end]
            if(len(batch_data) != self.BATCH_SIZE):
                break
            note_list = []
            for j in range(len(batch_data[0])):
                batch_dataset = np.zeros([self.BATCH_SIZE, len(self.data_to_int)])
                for k in range(self.BATCH_SIZE):
                    note = batch_data[k][j]
                    idx = self.data_to_int[note]
                    batch_dataset[k, idx] = 1
                note_list.append(batch_dataset)
            train_dataset.append(note_list)
            self.train_data = train_dataset
        return train_dataset




