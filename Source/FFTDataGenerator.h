#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include "AudioFFT/AudioFFT.h"

class FFTDataGenerator
{
public:
    int fftSize;

    FFTDataGenerator(int _fftSize, int _sampleRate);

    void reassignedSpectrogram(
        juce::AudioBuffer<float>& buffer,
        std::vector<float>& times,
        std::vector<float>& frequencies,
        std::vector<float>& magnitudes,
        std::vector<float>& standardFFTResult,
        std::vector<float>& ncResult
    );

    void updateTimeWeightedWindow();

    void updateDerivativeWindow();

    void updateDerivativeTimeWeightedWindow();

    std::vector<std::complex<float>> doFFT(
        const juce::AudioBuffer<float>&inputBuffer, 
        std::vector<float>& window
    );

    std::vector<std::complex<float>> doFFTNoWindow(const juce::AudioBuffer<float>&inputBuffer);

    void resizeIfNecessary(std::vector<float>& vector, int size);

    void updateParameters(float despeckleCutoff, float fftSize);

private:
    int sampleRate;
    std::vector<float> standardWindow;
    std::vector<float> derivativeWindow;
    std::vector<float> timeWeightedWindow;
    std::vector<float> derivativeTimeWeightedWindow;
    juce::dsp::FFT fft;
    audiofft::AudioFFT ncfft;
    float despecklingCutoff;
};