#pragma once
#include "FFTDataGenerator.h"
#include "juce_dsp/juce_dsp.h"

FFTDataGenerator::FFTDataGenerator(int _fftSize, int _sampleRate):
    fftSize(_fftSize),
    sampleRate(_sampleRate),
    standardWindow(_fftSize, 0.0f),
    derivativeWindow(_fftSize, 0.0f),
    timeWeightedWindow(_fftSize, 0.0f),
    derivativeTimeWeightedWindow(_fftSize, 0.0f),
    fft(std::log2(_fftSize)),
    despecklingCutoff(2.f)
{
    // ncfft.init(fftSize * 2);
    ncfft.init(_fftSize);
    updateParameters(_fftSize, 2.f);
}

static int IdxWrap(int x, int length) {
    return (x % length + length) % length;
}

void FFTDataGenerator::reassignedSpectrogram(
    juce::AudioBuffer<float>& buffer,
    std::vector<float>& times,
    std::vector<float>& frequencies,
    std::vector<float>& magnitudes,
    std::vector<float>& standardFFTResult,
    std::vector<float>& ncResult
) {
    int bufferSize = buffer.getNumSamples();
    auto spectrumHann = doFFT(buffer, standardWindow);
    auto spectrumRect = doFFTNoWindow(buffer);

    auto spectrumHannDerivative = doFFT(buffer, derivativeWindow);
    auto spectumHannTimeWeighted = doFFT(buffer, timeWeightedWindow);
    auto spectrumHannDerivativeTimeWeighted = doFFT(buffer, derivativeTimeWeightedWindow);

    float currentFrequency = 0.f;
    float frequencyCorrectionRadians = 0.f;
    float frequencyCorrectionHz = 0.f;
    float correctedTimeSeconds = 0.f;
    float magnitude = 0.f;
    float currentTimeSeconds = 0.f;
    float fftBinSize = (float)sampleRate / (float)fftSize;
    float pi = 3.14159265358979;
    float totalTimeSeconds = (float)bufferSize / (float)sampleRate;
    float mixedPartialPhaseDerivative = 0.f;
    float magnitudeSquared = 0.f;

    std::complex<float> X;
    std::complex<float> X_Dh;
    std::complex<float> X_Th;
    std::complex<float> X_T_Dh;
    std::vector<float> mixedDerivative(fftSize / 2, 0.f);

    resizeIfNecessary(times, fftSize / 2);
    resizeIfNecessary(frequencies, fftSize / 2);
    resizeIfNecessary(magnitudes, fftSize / 2);
    resizeIfNecessary(standardFFTResult, fftSize / 2);
    resizeIfNecessary(ncResult, fftSize / 2);

    for (int bin = 0; bin < fftSize / 2; ++bin) {
        auto& thisBin = spectrumRect[bin];
        auto& nextBin = spectrumRect[bin + 1];
        float ncSum = -(thisBin.real() * nextBin.real() + thisBin.imag() * nextBin.imag());
        if (ncSum < 0) {
            ncResult[bin] = -100.0f;
        }
        else {
            float gain = std::sqrt(ncSum + 1e-18f);
            ncResult[bin] = 20.0f * std::log10(gain);
        }
    }

    for (int frequencyBin = 0; frequencyBin < fftSize / 2; frequencyBin++) {
        currentFrequency = frequencyBin * fftBinSize;
        X = spectrumHann[frequencyBin];
        X_Dh = spectrumHannDerivative[frequencyBin];
        X_Th = spectumHannTimeWeighted[frequencyBin];
        X_T_Dh = spectrumHannDerivativeTimeWeighted[frequencyBin];
        magnitude = std::abs(X);
        magnitudeSquared = magnitude * magnitude;

        standardFFTResult[frequencyBin] = juce::Decibels::gainToDecibels(2 * magnitude);
            
        if (magnitude == 0) {
            continue;
        }
        
        float t1 = std::real(X_T_Dh * std::conj(X) / magnitudeSquared);
        float t2 = std::real(X_Th * X_Dh / magnitudeSquared);

        mixedPartialPhaseDerivative = t1 - t2;
        mixedDerivative[frequencyBin] = mixedPartialPhaseDerivative;

        if (std::abs(mixedPartialPhaseDerivative) > despecklingCutoff) {
            magnitude = 0; // filter it out.
        }

        frequencyCorrectionRadians = -std::imag(X_Dh / X);
        frequencyCorrectionHz = frequencyCorrectionRadians * sampleRate / (2 * pi);
        correctedTimeSeconds = std::real(X_Th / X) / sampleRate;

        times[frequencyBin] = correctedTimeSeconds;
        frequencies[frequencyBin] = currentFrequency + frequencyCorrectionHz;

        // in Gain, such that a known reassigned sine wave at an amplitude of 1 gets a magnitude of 1.
        // I'm not sure where the other factor of 2 is coming from.
        magnitudes[frequencyBin] = juce::Decibels::gainToDecibels(2 * magnitude); 
    }
} 

void FFTDataGenerator::updateParameters(float _fftSize, float _despecklingCutoff) {
    despecklingCutoff = _despecklingCutoff;
    fftSize = _fftSize;
    fft = juce::dsp::FFT(std::log2(_fftSize));
    // ncfft.init(fftSize * 2);
    ncfft.init(fftSize);

    resizeIfNecessary(standardWindow, fftSize);
    juce::dsp::WindowingFunction<float>::fillWindowingTables(standardWindow.data(), _fftSize, juce::dsp::WindowingFunction<float>::blackmanHarris, false);
    updateTimeWeightedWindow();
    updateDerivativeWindow();
    updateDerivativeTimeWeightedWindow();
}

void FFTDataGenerator::updateTimeWeightedWindow() {
    resizeIfNecessary(timeWeightedWindow, fftSize);

    int halfWidth = fftSize / 2;
    int index = 0;

    float maxValue = 0.f;

    // The center should be time = 0.
    for (int time = -halfWidth; time < halfWidth; time++) {
        index = time + halfWidth;
        float newValue = standardWindow[index] * time;
        timeWeightedWindow[index] = newValue;

        if (timeWeightedWindow[index] > maxValue) {
            maxValue = newValue;
        }
    }

    // There should be some math to calculate it without having to normalize the values.
    for (int i = 0; i < fftSize; i++) {
        timeWeightedWindow[i] = timeWeightedWindow[i] / maxValue;
    }
}

void FFTDataGenerator::updateDerivativeWindow() {
    resizeIfNecessary(derivativeWindow, fftSize);

    for (int i = 0; i < fftSize; i++)
    {
        if (i == 0 || i == fftSize - 1) {
            derivativeWindow[i] = 0;
            continue;
        }

        derivativeWindow[i] = (standardWindow[i + 1] - standardWindow[i - 1]) / 2.0;
    }
}

void FFTDataGenerator::updateDerivativeTimeWeightedWindow() {
    resizeIfNecessary(derivativeTimeWeightedWindow, fftSize);

    for (int i = 0; i < fftSize; i++)
    {
        derivativeTimeWeightedWindow[i] = derivativeWindow[i] * i;
    }
}

void FFTDataGenerator::resizeIfNecessary(std::vector<float>& vector, int size) {
    if (vector.size() != size) {
        vector.resize(size, 0.f);
    }
}

std::vector<std::complex<float>> FFTDataGenerator::doFFT(const juce::AudioBuffer<float>& inputBuffer, std::vector<float>& window) {
    std::vector<std::complex<float>> frame(fftSize, 0.0f);
    std::vector<std::complex<float>> fftResult(fftSize, 0.0f);
    
    const float* inputChannelData = inputBuffer.getReadPointer(0);

    for (int i = 0; i < fftSize; i++) {
        frame[i] = inputChannelData[i] * window[i];
    }

    // Perform FFT
    fft.perform(frame.data(), fftResult.data(), false);

    // Normalize the values by the FFT size, and multiply by 2 
    // because we are splitting the energy between positive and negative frequencies.
    for (int i = 0; i < fftSize; i++) {
        fftResult[i] /= (fftSize / 2);
    }

    return fftResult;
}

std::vector<std::complex<float>> FFTDataGenerator::doFFTNoWindow(const juce::AudioBuffer<float>& inputBuffer) {
    std::vector<float> frame(fftSize, 0.0f);
    std::vector<float> real(fftSize / 2 + 1, 0.0f);
    std::vector<float> imag(fftSize / 2 + 1, 0.0f);
    std::vector<std::complex<float>> fftResult(fftSize + 1, 0.0f);
    
    const float* inputChannelData = inputBuffer.getReadPointer(0);
    for (int i = 0; i < fftSize; i++) {
        frame[i] = inputChannelData[i];
    }

    // Perform FFT
    // ncFft.perform(frame.data(), fftResult.data(), false);
    ncfft.fft(frame.data(), real.data(), imag.data());

    // Normalize the values by the FFT size, and multiply by 2 
    // because we are splitting the energy between positive and negative frequencies.
    float mul = 1.0f / (fftSize / 2.0f);
    for (int i = 0; i < fftSize / 2 + 1; i++) {
        fftResult[i].real(real[i] * mul);
        fftResult[i].imag(imag[i] * mul);
    }

    return fftResult;
}