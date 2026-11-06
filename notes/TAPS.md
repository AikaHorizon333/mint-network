# Number of Taps

**T = number of FIR taps** of your equalizer filter.

Formally:

[
g[n], \quad n = 0, 1, \dots, T-1
]

So if:

* **T = 1024**,
  your filter has 1024 coefficients.

---

# What “number of taps” actually means physically

An FIR filter is just a convolution:

[
y[n] = \sum_{k=0}^{T-1} g[k],x[n-k]
]

Each tap:

* Multiplies a delayed version of the input
* Contributes to shaping magnitude + phase response

---

# What T Controls (Very Important)

### 1️⃣ Frequency Resolution

Longer T → sharper frequency control.

Approximate frequency resolution:

[
\Delta f \approx \frac{fs}{T}
]

At **fs = 48kHz**:

| T    | Δf     |
| ---- | ------ |
| 512  | ~94 Hz |
| 1024 | ~47 Hz |
| 2048 | ~23 Hz |

If you're correcting room modes at 40–60 Hz,
512 taps may be too short.
1024 or 2048 gives better control.

---

### 2️⃣ Latency (If Linear Phase)

If you enforce symmetric FIR:

[
\text{group delay} = \frac{T-1}{2}
]

At 48 kHz:

| T    | Latency  |
| ---- | -------- |
| 512  | ~5.3 ms  |
| 1024 | ~10.7 ms |
| 2048 | ~21.3 ms |

For:

* Studio mixing → 10 ms is fine
* Real-time instrument monitoring → maybe too much
* Offline rendering → irrelevant

---

### 3️⃣ Computational Cost

Longer T → more CPU:

* Direct convolution: O(T)
* Partitioned FFT: O(log T)

For 1024 taps at 48kHz, this is trivial on modern CPUs.

---

# In Your Specific Case (436 ms IR)

Your IR length = **20944 samples (~436 ms)**

That does NOT mean T must be 20944.

The equalizer:

* Does not need to model the entire room decay
* Only needs enough taps to shape frequency response in-band

For most room EQ systems:

* 512–2048 taps is typical
* 1024 is a very solid starting point

---

# Conceptual Relationship

You measured:

[
H(f)
]

You want to design:

[
G(f)
]

So that:

[
H(f)G(f) \approx 1
]

But since deep nulls are not invertible,
T controls **how precisely** you can approximate the inverse.

---

# Big Engineering Insight

Increasing T does NOT magically fix:

* deep cancellations
* noisy high frequencies
* modal nulls

It only:

* increases frequency resolution
* increases time support

That’s why regularization is more important than blindly increasing T.

---

# For Your Neural EQ Implementation

We are literally training the network to output:

```
g ∈ ℝ^T
```

So yes:

* T = output dimension of the model
* T = number of trainable taps (directly or indirectly)

---

If you tell me:

* Is this for speakers in a room?
* Car cabin?
* Headphones?
* Studio monitors?

I can tell you the most rational T to start with — not just a generic recommendation.
