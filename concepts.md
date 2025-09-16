## Individual Diagrams

### Slide 2: Traditional PINN vs Our Approach

```mermaid
graph TD
    A[New Physics Problem] --> B[Traditional PINN]
    B --> C[Train from Scratch]
    C --> D[500 Steps]
    D --> E[Solution for Single Problem]
    
    A --> F[Our PI-MAML]
    F --> G[Meta-Learned Parameters]
    G --> H[Fast Adaptation]
    H --> I[50 Steps]
    I --> J[Solution + Knowledge Transfer]
    
    style B fill:#ffcccc
    style F fill:#ccffcc
    style C fill:#ffcccc
    style H fill:#ccffcc
```

### Slide 3: Motivating Example - Fluid Dynamics

```mermaid
graph LR
    subgraph "Traditional Approach"
        Re100[Re = 100<br/>Train 500 steps]
        Re200[Re = 200<br/>Train 500 steps]
        Re500[Re = 500<br/>Train 500 steps]
        Re1000[Re = 1000<br/>Train 500 steps]
    end
    
    subgraph "Our Approach"
        Meta[Meta-Learning<br/>Across All Re]
        Adapt[Fast Adaptation<br/>50 steps per Re]
        Transfer[Knowledge Transfer<br/>Between Problems]
    end
    
    subgraph "Shared Physics"
        NS[Navier-Stokes]
        Momentum[Momentum Conservation]
        Continuity[Mass Conservation]
    end
    
    Re100 --> Meta
    Re200 --> Meta
    Re500 --> Meta
    Re1000 --> Meta
    
    Meta --> Adapt
    Meta --> Transfer
    
    NS --> Transfer
    Momentum --> Transfer
    Continuity --> Transfer
```

### Few-Shot Parameter Inference Explanation

```mermaid
graph TB
    subgraph "Traditional ML"
        TML[Standard Supervised Learning]
        TD[Thousands of Data Points]
        NP[No Physics Knowledge]
    end
    
    subgraph "Few-Shot Challenge"
        FS[Few-Shot Learning]
        FD[5-100 Data Points Only]
        PI[Infer Physical Parameters]
        Examples[Re, viscosity, boundary conditions]
    end
    
    subgraph "Our Solution"
        ML[Meta-Learning]
        PC[Physics Constraints]
        FA[Fast Adaptation]
        PK[Prior Knowledge Transfer]
    end
    
    TML --> FS
    TD --> FD
    FD --> PI
    PI --> Examples
    
    FS --> ML
    PI --> PC
    ML --> FA
    PC --> PK
```

### Slide 4: Meta-Learning Concept

```mermaid
graph LR
    subgraph "Training Phase"
        T1[Task 1: Re=100]
        T2[Task 2: Re=200]
        T3[Task 3: Re=500]
        T4[Task N: Re=1000]
    end
    
    subgraph "Meta-Learning"
        ML[Learn to Learn]
        MP[Meta-Parameters θ]
    end
    
    subgraph "New Task"
        NT[Re=300]
        FA[Fast Adaptation]
        S[Solution in 50 steps]
    end
    
    T1 --> ML
    T2 --> ML
    T3 --> ML
    T4 --> ML
    ML --> MP
    MP --> FA
    NT --> FA
    FA --> S
```

### Slide 5: Framework Overview

```mermaid
graph TB
    subgraph "Input"
        TD[Task Distribution p T]
        PD[Physics Domain Omega]
        BC[Boundary Conditions]
    end
    
    subgraph "Our Framework"
        AML[Physics-Informed Meta-Learning]
        ACW[Adaptive Constraint Weighting]
        APD[Automated Physics Discovery]
    end
    
    subgraph "Outputs"
        MP[Meta-Parameters]
        FA[Fast Adaptation]
        PI[Physics Insights]
    end
    
    TD --> AML
    PD --> AML
    BC --> AML
    AML --> ACW
    AML --> APD
    ACW --> MP
    APD --> PI
    MP --> FA
```

### Slide 6: Problem Formulation

```mermaid
graph TB
    subgraph "Task Ti"
        Domain[Domain Ωi ⊂ ℝᵈ]
        Boundary[Boundary ∂Ωi]
        PDE[PDE: Fi[ui](x) = 0]
        BC[BC: Bi[ui](x) = 0]
        Data[Data: Di = {(xj, uj)}]
    end
    
    subgraph "Task Distribution"
        T1[Task 1]
        T2[Task 2]
        TN[Task N]
    end
    
    subgraph "Meta-Learning Goal"
        MetaModel[Meta-Model θ]
        FastAdapt[Fast Adaptation φT]
        Physics[Respect Physics]
    end
    
    Domain --> PDE
    Boundary --> BC
    Data --> FastAdapt
    
    T1 --> MetaModel
    T2 --> MetaModel
    TN --> MetaModel
    
    MetaModel --> FastAdapt
    FastAdapt --> Physics
```

### Slide 7: Physics-Informed Meta-Learning Algorithm

```mermaid
graph TD
    Start[Initialize θ] --> SB[Sample Task Batch]
    SB --> IT[For each Task Ti]
    IT --> TE[Compute Task Embedding hTi]
    TE --> AW[Adaptive Weight λi = σ(WλhTi + bλ)]
    AW --> IP[Initialize φi = θ]
    IP --> IL[Inner Loop: K steps]
    
    subgraph "Inner Loop"
        CP[Sample Collocation Points]
        DL[Compute Data Loss]
        PL[Compute Physics Loss: |F[uφ]|²]
        TL[Total Loss: Ldata + λiLphysics]
        UP[Update: φi = φi - β∇φiLtotal]
    end
    
    IL --> CP
    CP --> DL
    DL --> PL
    PL --> TL
    TL --> UP
    UP --> QE[Evaluate on Query Set]
    QE --> MG[Compute Meta-Gradient]
    MG --> UM[Update Meta-Parameters θ]
    UM --> Conv{Converged?}
    Conv -->|No| SB
    Conv -->|Yes| End[Return θ, Wλ, bλ]
```

### Slide 8: Physics Loss Implementation

```mermaid
graph TB
    subgraph "Domain Ω"
        IP[Interior Points]
        BP[Boundary Points ∂Ω]
    end
    
    subgraph "Physics Loss"
        PDE[PDE Residual: |F[uφ](x)|²]
        BCs[Boundary Conditions: |B[uφ](x)|²]
        PL[Lphysics = E[PDE] + E[BCs]]
    end
    
    subgraph "Implementation"
        AD[Automatic Differentiation]
        CM[Collocation Method]
        MC[Monte Carlo Sampling]
    end
    
    IP --> PDE
    BP --> BCs
    PDE --> PL
    BCs --> PL
    PL --> AD
    AD --> CM
    CM --> MC
```

### Slide 9: Adaptive Constraint Weighting

```mermaid
graph LR
    subgraph "Task Characteristics"
        TC[Task Ti]
        GE[Geometry]
        PC[Physics Complexity]
        DP[Data Points]
    end
    
    subgraph "Task Encoder"
        NN[Neural Network]
        TE[Task Embedding hTi]
    end
    
    subgraph "Adaptive Weighting"
        AW[λ(T) = σ(WλhT + bλ)]
        Lambda[Physics Weight λi]
    end
    
    TC --> NN
    GE --> NN
    PC --> NN
    DP --> NN
    NN --> TE
    TE --> AW
    AW --> Lambda
```

### Slide 12: Experimental Setup Overview

```mermaid
graph TB
    subgraph "Problem Classes"
        NS[Navier-Stokes Equations<br/>Re: 100-1000]
        HT[Heat Transfer<br/>Varying BCs]
        BE[Burgers Equation<br/>Different Viscosity]
        LC[Lid-Driven Cavity<br/>Varying Geometry]
    end
    
    subgraph "Dataset Structure"
        Train[200 Training Tasks]
        Test[50 Test Tasks]
        Adapt[20-100 Data Points/Task]
        Few[Few-Shot Scenarios]
    end
    
    subgraph "Evaluation"
        Acc[Accuracy Metrics]
        Eff[Efficiency Analysis]
        Stat[Statistical Testing]
        Abl[Ablation Studies]
    end
    
    NS --> Train
    HT --> Train
    BE --> Train
    LC --> Train
    
    Train --> Test
    Test --> Adapt
    Adapt --> Few
    
    Few --> Acc
    Few --> Eff
    Acc --> Stat
    Eff --> Abl
```

### Slide 20: Limitations - Domain Specificity

```mermaid
graph TB
    subgraph "Current Scope"
        FD[Fluid Dynamics Focus]
        PR[Parameter Ranges Limited]
        TD[Tested Domains Only]
    end
    
    subgraph "Limitations"
        BS[Broader Scope Needed]
        HD[High-Dimensional Unclear]
        TS[Task Similarity Assumptions]
        RA[Regularity Assumptions]
    end
    
    subgraph "Constraints"
        SM[Smoothness Requirements]
        CS[Collocation Sampling]
        AD[Auto-Diff Limitations]
    end
    
    FD --> BS
    PR --> HD
    TD --> TS
    
    BS --> RA
    HD --> SM
    TS --> CS
    RA --> AD
```

### Slide 21: Limitations - Theoretical Assumptions

```mermaid
graph TB
    subgraph "Mathematical Assumptions"
        LC[Lipschitz Continuity]
        BV[Bounded Variance]
        SC[Strong Convexity]
        RC[Regularity Conditions]
    end
    
    subgraph "Practical Issues"
        DA[94% Discovery Accuracy]
        TE[Task Embedding Quality]
        FS[Finite Sample Effects]
        VR[Violation of Regularity]
    end
    
    subgraph "Research Areas"
        IA[Improved Accuracy]
        BTE[Better Task Encoding]
        FSA[Few-Shot Analysis]
        RA[Relaxed Assumptions]
    end
    
    LC --> DA
    BV --> TE
    SC --> FS
    RC --> VR
    
    DA --> IA
    TE --> BTE
    FS --> FSA
    VR --> RA
```

### Slide 22: Future Work - Broader Physics Domains

```mermaid
graph TB
    subgraph "Current Domain"
        CFD[Computational Fluid Dynamics]
        NS[Navier-Stokes]
        HT[Heat Transfer]
        BE[Burgers Equation]
    end
    
    subgraph "Target Domains"
        SM[Solid Mechanics]
        EM[Electromagnetics]
        QM[Quantum Mechanics]
        MP[Multi-Physics]
    end
    
    subgraph "Methodological Advances"
        HML[Hierarchical Meta-Learning]
        SAI[Symbolic AI Integration]
        UQ[Uncertainty Quantification]
        AL[Active Learning]
    end
    
    CFD --> SM
    NS --> EM
    HT --> QM
    BE --> MP
    
    SM --> HML
    EM --> SAI
    QM --> UQ
    MP --> AL
```

### Slide 23: Theoretical Extensions Roadmap

```mermaid
graph LR
    subgraph "Current Theory"
        CT[Convex Constraints]
        LB[Lipschitz Bounds]
        SC[Sample Complexity]
    end
    
    subgraph "Extensions"
        NCT[Non-Convex Theory]
        DS[Distribution Shift]
        MTB[Multi-Task Bounds]
        AT[Approximation Theory]
    end
    
    subgraph "Algorithms"
        SO[Second-Order Methods]
        GF[Gradient-Free]
        FL[Federated Learning]
        CL[Continual Learning]
    end
    
    CT --> NCT
    LB --> DS
    SC --> MTB
    NCT --> AT
    
    DS --> SO
    MTB --> GF
    AT --> FL
    SO --> CL
```

### Slide 24: Applications and Impact

```mermaid
graph LR
    subgraph "Resource-Constrained"
        RTC[Real-Time Control]
        EC[Edge Computing]
        RP[Rapid Prototyping]
        ER[Emergency Response]
    end
    
    subgraph "Scientific Discovery"
        PE[Parameter Estimation]
        MS[Model Selection]
        AD[Anomaly Detection]
        HG[Hypothesis Generation]
    end
    
    subgraph "Framework Benefits"
        FA[Fast Adaptation]
        LE[Low Energy]
        AI[Automated Insights]
        TG[Theoretical Guarantees]
    end
    
    FA --> RTC
    LE --> EC
    AI --> RP
    TG --> ER
    
    FA --> PE
    AI --> MS
    TG --> AD
    AI --> HG
```

### Slide 25: Before/After Comparison

```mermaid
graph LR
    subgraph "Before: Traditional PINNs"
        BR[Retrain for Each Problem]
        BT[500 Steps Required]
        BN[No Knowledge Transfer]
        BL[Limited Scalability]
    end
    
    subgraph "After: Our PI-MAML"
        AR[Fast Adaptation]
        AT[50 Steps Only]
        AK[Knowledge Transfer]
        AS[Scalable Framework]
    end
    
    subgraph "Quantitative Improvements"
        Acc[92.4% Accuracy]
        Imp[15% Improvement]
        Eff[3x Faster]
        Disc[94% Discovery]
    end
    
    BR --> AR
    BT --> AT
    BN --> AK
    BL --> AS
    
    AR --> Acc
    AT --> Eff
    AK --> Imp
    AS --> Disc
```

### Slide 26: Technical Contributions Summary

```mermaid
graph TD
    subgraph "Algorithmic Innovations"
        PIML[Physics-Informed Meta-Learning]
        ACW[Adaptive Constraint Weighting]
        APD[Automated Physics Discovery]
    end
    
    subgraph "Theoretical Advances"
        CRA[Convergence Rate Analysis]
        SCB[Sample Complexity Bounds]
        MG[Mathematical Guarantees]
    end
    
    subgraph "Experimental Validation"
        RSA[Rigorous Statistical Analysis]
        AS[Ablation Studies]
        MPD[Multiple Physics Domains]
    end
    
    PIML --> CRA
    ACW --> SCB
    APD --> MG
    
    CRA --> RSA
    SCB --> AS
    MG --> MPD
```

### Slide 27: Research Impact Visualization

```mermaid
graph TB
    subgraph "Meta-Learning Community"
        ML[Meta-Learning Research]
        FS[Few-Shot Learning]
        MAML[Model-Agnostic Methods]
    end
    
    subgraph "Physics-Informed ML"
        PINN[Physics-Informed Neural Networks]
        SciML[Scientific Machine Learning]
        CFD[Computational Physics]
    end
    
    subgraph "Our Contribution"
        Framework[PI-MAML Framework]
        Theory[Theoretical Foundation]
        Applications[Practical Applications]
    end
    
    subgraph "Future Impact"
        NewDomains[New Physics Domains]
        AIScience[AI for Science]
        AutoDiscovery[Automated Discovery]
    end
    
    ML --> Framework
    PINN --> Framework
    Framework --> Theory
    Framework --> Applications
    
    Theory --> NewDomains
    Applications --> AIScience
    Framework --> AutoDiscovery
```
