    Learnable Multi-hierarchical Threshold Leaky Integrate-and-Fire Neuron theory
    with Hybrid Reset mechanism and Leaky Memory

    Key Formulas:

    1. Membrane Potential Update:
       m^l(t) = λ^l * v^l(t-1) + (1-λ^l) * I^l(t)
       where:
       - m^l(t): membrane potential before reset at time t
       - λ^l: leakage factor (0 < λ^l ≤ 1)
       - v^l(t-1): membrane potential after reset at previous time step
       - I^l(t): input current at time t

    2. Input Current Calculation:
       I^l(t) = W^l · s^(l-1)(t) · θ^(l-1)
       where:
       - W^l: weight matrix for layer l
       - s^(l-1)(t): spike output from layer l-1 at time t
       - θ^(l-1): threshold of layer l-1

    3. Threshold Levels:
       θ_k = k * (θ_max / K)
       where:
       - θ_k: k-th threshold level
       - θ_max: maximum threshold value
       - K: total number of threshold levels
       - k ∈ {1, 2, ..., K}

    4. Spike Generation:
       s^l(t) = {
           k,  if θ_(k-1) ≤ m^l(t) < θ_k
           0,  otherwise
       }
       where k is the highest threshold crossed

    5. Hybrid Reset Mechanism:
       v^l(t) = {
           0,           if s^l(t) = K (max threshold)
           m^l(t) - k*θ_base,  if 0 < s^l(t) < K (intermediate)
           m^l(t),      if s^l(t) = 0 (no spike)
       }
       where θ_base = θ_max / K is the base threshold unit
