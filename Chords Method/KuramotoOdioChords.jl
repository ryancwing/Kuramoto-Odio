##
## Simulation of the Kuramoto coupled oscillator model on a 2D grid
## By: Christian Koertje
## MIDI Audio data generation from model data 
## By: Ryan Wing
## Last updated: 05/06/2022
##
## updated the record function

using GLMakie
using ProgressBars
using Distances
using FFTW
using MIDI
using Statistics

## Spatial parameters
n = Int64(128)

## Temporal parameters
Δt = Float64(0.01)
T = Float64(Δt*15000) #Float64(100.0)

## Model parameters
w = 10 # base frequency for each oscillator
dw = 0 # magnitude of random perturbation to w
K = Float64(1.5) # coupling constant

winSize = 10

## construct containers for state variables
function initialize(duration)
    global θ = Array{Float64, 2}(2*π * rand(n, n))
    global nextθ = Array{Float64, 2}(zeros(n, n))
    global ω = Array{Float64, 2}(w * ones(n, n) + dw*rand(n, n))
    global θHist = Array{Float64, 3}(undef, n, n, duration)

    global stateθ = Observable{Array{Float64, 2}}()
end

## observes state varibales for live and recorded plot
function observe()
    global θ = θ
    global stateθ = stateθ
    stateθ[] = θ
end

## plots the heatmap of the phase θ with a colorbar    
function plotState()
    global stateθ = stateθ
    global fig = Figure(resolution = (800, 500))

    ax = Axis(fig[1,1])
    hm = heatmap!(ax, stateθ, colormap=:lightrainbow)
    Colorbar(fig[1,2], hm)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    resize_to_layout!(fig)
    display(fig)
end

## updates simulation by solving the system of differential 
## equations
function update(i)
    global θ, nextθ, ω, θHist = θ, nextθ, ω, θHist

    θHist[:,:,i] = θ
    # display(θ)

    # get neighborhood
    nbs = Vector{Array{Float64, 2}}()
    for dx ∈ -1:1
        for dy ∈ -1:1
            push!(nbs, 1 * circshift(θ, (dy, dx)))
        end
    end

    # forward-euler
    nextθ = θ + (Δt) * ω + (Δt * K) * sum(sin.(nbs[i] - θ) for i ∈ 1:length(nbs))

    θ, nextθ = mod.(nextθ, 2*π), θ
end


## MIDI

function micmap_gen()
    
    # Generates a matrix of distance-to-center values for each quadrant of 128x128 model
    # To be used to attenuate amplitude of each oscillator proximity to "microphone" at center of each quadrant

    global quadbase = Array{Float64, 2}(zeros(Int(n/2),Int(n/2)))
    for i in 1:Int(size(quadbase, 1))
        for j in 1:Int(size(quadbase, 2))
            quadbase[i,j] = 1-(euclidean((i,j), (Int(n/4), Int(n/4)))/127)
        end
    end
    fullrow = hcat(quadbase,quadbase)
    global quadmicmap = vcat(fullrow,fullrow)
end

function plotMicMap()
    global fig = Figure(resolution = (800, 500))

    ax = Axis(fig[1,1])
    hm = heatmap!(ax, quadmicmap, colormap=:lightrainbow)
    Colorbar(fig[1,2], hm)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    resize_to_layout!(fig)
    display(fig)
end

function midi_setup()
    
    # ticks/pulses per quarter note (tpq/ppq) conversion to seconds
    # 60 / (BPM * PPQ) (seconds)
    global bpm = Int(120)
    global set_tpq = Int(1000)
    global one_tick = 60 / (bpm * set_tpq) # in seconds
    global tpts = Int16(Δt / one_tick)
    
    # setup MIDINotes objects for each quadrant
    global mn1 = Notes(tpq=set_tpq)
    global mn2 = Notes(tpq=set_tpq)
    global mn3 = Notes(tpq=set_tpq)
    global mn4 = Notes(tpq=set_tpq)
    
end

function quad_split(array)
    global q1 = Array{Float64, 2}(zeros(Int(n/2),Int(n/2)))
    global q2 = Array{Float64, 2}(zeros(Int(n/2),Int(n/2)))
    global q3 = Array{Float64, 2}(zeros(Int(n/2),Int(n/2)))
    global q4 = Array{Float64, 2}(zeros(Int(n/2),Int(n/2)))
    for i in 1:Int((size(array, 1)/2))
        for j in 1:Int((size(array, 2)/2))
            q1[i,j] = array[i,j]
        end
    end
    for i in 1:Int((size(array, 1)/2))
        for j in 65:Int(size(array, 2))
            q2[i,j-Int(n/2)] = array[i,j]
        end
    end
    for i in 65:Int(size(array, 1))
        for j in 1:Int((size(array, 2)/2))
            q3[i-Int(n/2),j] = array[i,j]
        end
    end
    for i in 65:Int(size(array, 1))
        for j in 65:Int(size(array, 2))
            q4[i-Int(n/2),j-Int(n/2)] = array[i,j]
        end
    end
end

function midi_gen(quad, output)

    # temp storage for timestep note velocities
    Cv = []
    Ev = []
    Gv = []
    Bv = []
    full_vel = sum(quadbase)
    # use theta value ranges to classify notes
    # pull velocity values from quadmicmap for each note
    # add velocity value to cooresponding note velocity list
    for i in 1:size(quad, 1)
        for j in 1:size(quad, 2)
            if 0.0 < quad[i,j] <= (pi/2)
                push!(Cv, quadmicmap[i,j])
            elseif pi/2 < quad[i,j] <= pi
                push!(Ev, quadmicmap[i,j])
            elseif pi < quad[i,j] <= pi*1.5
                push!(Gv, quadmicmap[i,j])
            else pi*1.5 < quad[i,j] <= pi*2
                push!(Bv, quadmicmap[i,j])
            end
        end
    end

    # calculate average velocities for each note
    Cvel = Int8(round((sum(Cv)/full_vel)*127))
    Evel = Int8(round((sum(Ev)/full_vel)*127))
    Gvel = Int8(round((sum(Gv)/full_vel)*127))
    Bvel = Int8(round((sum(Bv)/full_vel)*127))

    # notes dynamically created from parameters
    # do we need to specify channel numbers for different quads?
    C = Note(48, Cvel, ticktock, tpts)
    E = Note(52, Evel, ticktock, tpts)
    G = Note(67, Gvel, ticktock, tpts)
    B = Note(71, Bvel, ticktock, tpts)

    # add each note to Notes object cooresponding to quadrant number
    push!(output, C, E, G, B)

end


function midi_output()
    save("KuramotoOdioQ1Chords.mid", mn1)
    save("KuramotoOdioQ2Chords.mid", mn2)
    save("KuramotoOdioQ3Chords.mid", mn3)
    save("KuramotoOdioQ4Chords.mid", mn4)
end



# Main
function main()
    timeFrame = Δt:Δt:T
    time = ProgressBar(timeFrame)
    global elapsed = Δt
    winPos = 1
    midi_setup()
    micmap_gen()
    print("Begin simulation... \n")
    initialize(length(timeFrame))
    observe()
    plotState()

    ## use this to record the simulation
    record(fig, "KuramotoOdioChords.mp4", enumerate(time);
        framerate = 100) do (i, t)
            global ticktock = Int(round(elapsed / one_tick))
            update(i)
            quad_split(θ)
            midi_gen(q1, mn1)
            midi_gen(q2, mn2)
            midi_gen(q3, mn3)
            midi_gen(q4, mn4)
            observe()
            elapsed += Δt
        end

    ## loop for simulation without recording (faster)
    # for (i,t) ∈ enumerate(time)
    #     global ticktock = Int(round(elapsed / one_tick))
    #     update(i)
    #     quad_split(θ)
    #     midi_gen()
    #     observe()
    #     elapsed += Δt
    # end
    midi_output()
    print("Simulation Complete! \n")
end


main()


# Finish / dial-in kuramoto model code

# Unite with video of visual model