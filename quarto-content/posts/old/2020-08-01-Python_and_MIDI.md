---
aliases:
- /markdown/2020/08/01/Python_and_MIDI
categories:
- music
date: '2020-08-01'
description: A minimal example of using markdown with fastpages.
layout: post
title: Making Music with pretty_midi
toc: true

---

# Making Music with pretty_midi

## MIDI Music

A while back, when I was first learning Python, a friend and I made a program that generated random music by iterating through a song one note at a time randomly picking the note length and pitch for the notes with increasingly strict rules. My friend figured out how to use [MIDIUtil](https://pypi.org/project/MIDIUtil/) to create a midi file by writing down a sequence of note events, where each event says something about:

* velocity - how loud the note should be played
* pitch - an integer value for telling a midi play what frequency to play
* note start - when the note should start playing
* note end - when it should stop

You can do a lot of music with just that information, and if you make simple choices for notes, with the right MIDI player you can get surprisingly listenable music. I have wanted to expand on this program for a while in various ways:

* generating music by applying machine learning techniques to various collections of midi files (like this [Google Bach doodle](https://www.google.com/doodles/celebrating-johann-sebastian-bach))
* making something more interactive (something involving the console where you can add motifs and ideas on the play and have them played back)
* feed the program just a text file with a simplistic music notation to easily create ideas that can get more complex (like [TidalCycles](https://tidalcycles.org/index.php/Welcome) which uses Haskell and SuperCollider).

## Creating MIDI Files with Python

I searched around on pypi a bit and found a promising package to start writing MIDI files with: [pretty_midi](https://pypi.org/project/pretty_midi/). It's a [project on GitHub](https://github.com/craffel/pretty-midi) that seems to still be active (part of why I am not just using MIDIUtil again is that I remember I slightly annoying setup and it seems to be inactive). After a simple install, and running the second example [from the documentation](http://craffel.github.io/pretty-midi/) I found that it easily generated pleasing sounds that (after a bit of fitzing) I could simply listen to with VLC Media Player. Before I just imported the files into Reaper where I could use a custom VST, but this makes iterating a bit quicker.

Here is an example I made where two instruments are playing a very minimalist piece:

```python
import pretty_midi

# Create a PrettyMIDI object
ensemble = pretty_midi.PrettyMIDI()

# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)

# do the same for a piano
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)

# Add the instruments to the PrettyMIDI object
ensemble.instruments.append(cello)
ensemble.instruments.append(piano)


song_length_in_seconds = 30
bpm = 120
beat_length = 60 / bpm
num_beats = int(song_length_in_seconds / beat_length)

for beat in range(num_beats):
    if beat % 2 == 0:
        note_name = 'C5'
    else:
        note_name = 'D5'
    note_number = pretty_midi.note_name_to_number(note_name)
    note_start = beat * beat_length
    note_end = note_start + beat_length
    note = pretty_midi.Note(velocity=100, pitch=note_number, start=note_start, end=note_end)
    cello.notes.append(note)
    piano.notes.append(note)


def write_song():
    # Write out the MIDI data
    ensemble.write('ensemble.mid')

if __name__ == "__main__":
    write_song()
```

This is a simple example, yet it shows the basics of the process that I will expand on:

* add notes one at a time
* incorporate some randomness into the note properties (here pitch changes deterministically, but that will soon change)
* using multiple instruments

I will begin to add to this, hopefully reaching what we achieved during my first experience with Python and MIDI and then think about ways to expand it and then start to apply some of my fastbook reading to it.
