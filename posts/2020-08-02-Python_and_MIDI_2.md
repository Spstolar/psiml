---
aliases:
- /markdown/2020/08/02/Python_and_MIDI_2
categories:
- markdown
date: '2020-08-02'
description: A minimal example of using markdown with fastpages.
layout: post
title: Make More Interesting Random Music
toc: true

---

# Make More Interesting Random Music

We saw last time that it is simple to construct a MIDI file with the `pretty_midi` package. Now to make something a little more musically interesting than alternating between two notes we need to randomly generate notes to play using some basic music theory to make things sound "good." To do this we will:

* choose a scale
* find all of the midi notes attached to that scale
* randomly draw notes from that collection of midi notes
* get a pleasing collection of simple chords to accompany the melody
* vary the rhythm of the melody

## Pick a Scale

First we use a couple of nice utilities of `pretty_midi`, a method which takes an integer and spits out one of the major and minor keys and another method which takes that same integer and returns how many accidentals the key has. Using the key name we can determine whether the key has flat accidentals or sharp accidentals and then by using the fact that "accidentals accumulate" (if a key has a G# then it also has a C#, for example) we can easily identify which notes are in the key.

To say more, we are using the fact that the Major and Minor keys have certain nice patterns. If we start from C and move up in perfect 5ths (C, G, D, A) then those corresponding (major) keys each add an accidental:

* C Major: CDEFGAB
* G Major: GABCDEF#
* D Major: DEF#GABC#
* A Major: ABC#DEF#G#
* ...

This means we know the exact notes that are made sharp or (flat) if we know if the key is sharp (flat) and how many accidentals there are, we do not even need to know the root: we are being told the same information in a different way. Now, there is actually a more insightful way to do this (how making major scales is normally taught) by using the fact that you start at the root and add notes with the pattern WWHWWWH, but it was fun to think of a different way to get the notes.

We return the `key_notes` which runs from C to B because this is how the MIDI format is laid out for each octave (... B1 C2 C#2 ... A#2 B2 C3 ...), but we also save off the notes of the key starting from the root for (optional) printing to the user as `scale_notes`.

```python
note_names = [n for n in 'CDEFGAB']
sharp_accidentals = [n for n in 'FCGDAEB']
flat_accidentals = [n for n in 'BEADGCF']

def determine_key_notes(key_number):
    key_name = pretty_midi.key_number_to_key_name(key_number)
    _, num_accidentals = pretty_midi.key_number_to_mode_accidentals(key_number)
    root = key_name[0]
    root_index = note_names.index(root)

    if key_name[1] == "b":
        accidental_mark = 'b'
        accidentals = flat_accidentals[:num_accidentals]
    else:
        accidental_mark = '#'
        accidentals = sharp_accidentals[:num_accidentals]

    key_notes = list(map(lambda n: n + accidental_mark if n in accidentals else n, note_names))

    scale_notes = (key_notes + key_notes)[root_index:root_index + 7]
```

## Find All MIDI Notes Given Key Notes

For this, use the `pretty_midi` utility that converts note names to MIDI note numbers and apply it to all the note names with all the octave numbers attached.

```python
def get_all_midi_numbers(note_names):
    midi_numbers = []
    note_names = list(set(note_names))  # simple de-dup
    for octave in range(-1, 9):
        for note in note_names:
            midi_number = pretty_midi.note_name_to_number(note + str(octave))
            midi_numbers.append(midi_number)
    return sorted(midi_numbers)
```

We do not need to de-duplicate or sort for our current use, but I added those steps in case I supply some note names "out of order" (`[D, C]` instead of `[C, D]` for instance) or provide possible note duplicates for other uses.

## Randomly Draw Notes

Here you can now use the collection of midi notes and just make a random choice from it at each step. For instance if `g_maj_midi` is the collection of MIDI notes for G Major then you can randomly select one with `pitch = random.choice(g_maj_midi[21:36])` where we take a slice of the array to restrict the notes to just a couple of octaves.

## Chord Accompaniment

We want to get just simple chords from the major scale for the piano to play them for whole notes. There are some existing Python packages that can be used:

* [chords2midi](https://github.com/Miserlou/chords2midi) let's you generate a MIDI file by supplying a progressing an a key: `c2m I V vi IV --key C`
* `chords2midi` uses [pychord](https://github.com/yuma-m/pychord), which will generate the component notes of a chord from its name as well as name a chord from its notes.

These are both strong utilities that I will certainly use as I expand but for now I will do something much simpler. You can get a major scale chord progression by simply going to the scale and taking a note for the root and getting the third and fifth of a triad by just taking the second and fourth notes after your root. For instance, you can get a C (major) triad from the C major scale by looking at the scale CDEFGAB and using that pattern C_E_G__. Whether this is major or minor is irrelevant for our use: we just want to grab all the simple triads (not worrying about inversions or anything) and collect them together for one octave of root notes:

```python
def get_major_progression(root_index, midi_numbers):
    chords = []
    for i in range(8):
        chord_root_index = root_index + i
        chord_third_index = chord_root_index + 2
        chord_fifth_index = chord_root_index + 4
        root = midi_numbers[chord_root_index]
        third = midi_numbers[chord_third_index]
        fifth = midi_numbers[chord_fifth_index]
        chord = [root, third, fifth]
        chords.append(chord)
    return chords
```

This will generate a list of chord lists, where each chord list is the midi notes for a given triad. Note, if you do not feed the notes of a key as the MIDI numbers you will not get a major key progression, so this might be imperfectly named.

## Vary the Rhythm

When we were adding notes one at a time we were keeping track of when the note began and when it ended in seconds. It would be nice to forget about when they begin and imagine writing the song and moving forward, adding notes "now." To do this we make a writer for our instrument that keeps track of when "now" is and allows us to add notes one at a time, as a chord, or in a chunk of notes. Using this chunking allows us to vary the rhythm, we can randomly determine how long the next note(s) should be and then play a bunch of notes of that length. This is slightly more natural than varying each note independently, because shorter notes are often grouped together.

```python
class Writer:
    def __init__(self, instrument):
        self.position = 0
        self.instrument = instrument

    def add_note(self, pitch, length, move_forward=True):
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=self.position, end=self.position + length)
        self.instrument.notes.append(note)

        if move_forward:
            self.position += length

    def add_note_series(self, pitches, length):
        for pitch in pitches:
            self.add_note(pitch, length)

    def add_chord(self, pitches, length, move_forward=True):
        for pitch in pitches:
            self.add_note(pitch, length, move_forward=False)
        if move_forward:
            self.position += length
```

We leave `move_forward` as an optional argument, because if we are writing notes that will occur at the same time (to form a chord) then we want the Writer "head" or position to stay at the same spot until we are done adding notes to that point in time. There are other simplifications that are made (no velocity changes), but this simple class gives us a lot of power so that we can more expressively generate midi music.

```python
import pretty_midi
from music_info import determine_key_notes
import music_info
import random

# Create a PrettyMIDI object
ensemble = pretty_midi.PrettyMIDI()

# Create an Instrument instance for a cello instrument
# Changed to a guitar for my song, and was lazy about changing variable names
# cello_program = pretty_midi.instrument_name_to_program('Cello')
cello_program = pretty_midi.instrument_name_to_program('Overdriven Guitar')
cello = pretty_midi.Instrument(program=cello_program)

# do the same for a piano
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)

# Add the instruments to the PrettyMIDI object
ensemble.instruments.append(cello)
ensemble.instruments.append(piano)

# here is where I put the Writer class from above

piano_writer = Writer(piano)
cello_writer = Writer(cello)

song_length_in_seconds = 30
bpm = 120
beat_length = 60 / bpm
num_beats = int(song_length_in_seconds / beat_length)

# Decided on G major, which is index 7
g_maj = music_info.determine_key_notes(7)
g_maj_midi = music_info.get_all_midi_numbers(g_maj)
root_number = pretty_midi.note_name_to_number("G4")
root_index = g_maj_midi.index(root_number)
major_progression = music_info.get_major_progression(root_index, g_maj_midi)

accompaniment_writer = piano_writer
solo_writer = cello_writer

# now we see the power of the writer object
while accompaniment_writer.position < song_length_in_seconds:
    length = beat_length * 4  # whole note chords

    # choose a random chord from the progression
    chord = random.choice(major_progression)  

    # add a lower octave of the notes for fullness
    larger_chord = chord + [n - 12 for n in chord]

    # write the chord
    accompaniment_writer.add_chord(larger_chord, length)

while solo_writer.position < song_length_in_seconds:
    # choose to play 16th, 8th, quarter, half, or whole note(s)
    division = random.choice([-2, -1, 0, 1, 2])
    length = beat_length * (2 ** division)

    # if we chose 16th or 8th, play multiple of them
    if division < 0:
        num_notes = 2 ** (-1 * division)
        pitches = random.choices(g_maj_midi[21:36], k=num_notes)
        solo_writer.add_note_series(pitches, length)
    else:
        pitch = random.choice(g_maj_midi[21:36])
        solo_writer.add_note(pitch, length=beat_length)

def write_song():
    # Write out the MIDI data
    ensemble.write('duo.mid')
```

Now with this I created the following [simple duet](https://soundcloud.com/simon-stolarczyk/duo). It isn't the most thrilling piece of music, but with relatively simple rules behind it, I think it is interesting how "complex" it sounds.
