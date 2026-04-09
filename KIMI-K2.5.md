



## Comment le code s'assemble et fonctionne

### 1. Chaîne de compilation

```bash
# Makefile simple (pas de CMake)
make                    # ou ./build.sh

# Étapes:
1. Compile C sources  →  .o (gcc -O3 -mavx2)
2. Compile ASM (NASM)  →  .o (nasm -f elf64)
3. Archive            →  libkmamba.a (ar rcs)
```

**Sources compilées:**

| Catégorie | Fichiers | Rôle |
|-----------|----------|------|
| **Core** | [src/kmamba.c](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/src/kmamba.c:0:0-0:0), [src/mamba_block.c](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/src/mamba_block.c:0:0-0:0) | API publique, forward/backward |
| **Topology** | `src/km_topology.c`, `src/wavefront_*.c` | ND indexing, wavefront |
| **Scan** | `src/scan_nd.c` | Référence ND |
| **Kernels** | `kernels/*.c` | GEMM, activations, optimizers |
| **ASM** | [cpu/scan1d.asm](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/cpu/scan1d.asm:0:0-0:0), [cpu/scan2d.asm](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/cpu/scan2d.asm:0:0-0:0) | Hot paths x86-64 |

---

### 2. Architecture d'exécution

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Utilisateur                    │
│              #include <kmamba.h> + libkmamba.a               │
├─────────────────────────────────────────────────────────────┤
│                      Layer 3: API Publique                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ kmamba.c ──► kmamba_forward(), kmamba_train_step()  │    │
│  │ mamba_block.c ──► mamba_block_forward(), backward() │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: Orchestration                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ km_topology.c ──► Normalisation ND des dimensions  │    │
│  │ wavefront_nd.c ──► Génération wavefront générique  │    │
│  │ wavefront_plan.c ──► Plans exécutables réutilisables│    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Compute Kernels                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ kernels/*.c     │  │ cpu/*.asm       │  │ cuda/*.cu   │  │
│  │ (inline C)      │  │ (AVX2/SSE)      │  │ (GPU)       │  │
│  │ • gemm_f32      │  │ • scan1d.asm    │  │ • scan1d.cu │  │
│  │ • activations   │  │ • scan2d.asm    │  │ • scan_nd.cu│  │
│  │ • optimizers    │  │                 │  │             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

### 3. Flux d'exécution typique (Forward)

```c
// Utilisateur:
KMamba *m = kmamba_create(&cfg);
kmamba_init(m, seed);
kmamba_forward(m, tokens, logits);
```

**Déroulement:**

```
kmamba_forward()
    ├── embed_lookup()                    # tokens → vecteurs [L,D]
    │
    ├── Pour chaque layer i:
    │   └── mamba_block_forward()
    │       ├── gemv_f32(W_in)            # x → z (projection entrée)
    │       ├── silu_f32(z)               # activation SiLU
    │       │
    │       ├── project_delta()           # calcule dt par position
    │       ├── gemv_f32(W_B) → B_t       # B sélectif
    │       ├── gemv_f32(W_C) → C_t       # C sélectif
    │       │
    │       └── scan1d()  ◄───┐           # === SSM CORE ===
    │           (dispatch)    │
    │               │         │
    │               ▼         │
    │       ┌─────────────┐   │
    │       │ cpu/scan1d  │   │  ou  │ cuda/scan1d.cu │
    │       │   .asm      │   │      │ (Blelloch GPU) │
    │       └─────────────┘   │
    │            │            │
    │            ▼            │
    │       h_t = exp(dt*A)*h_{t-1} + dt*B*x  # récurrence
    │       y_t = sum_m(C * h_t)              # sortie
    │                           │
    │       └───► retourne y    │
    │                           │
    ├── mb_matrix_vec_mult(head)             # projection finale
    └── softmax()                            # logits
```

---

### 4. Séparation des responsabilités

| Couche | Fichier clé | Rôle |
|--------|-------------|------|
| **Volontés** | [kmamba.c](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/src/kmamba.c:0:0-0:0), [mamba_block.c](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/src/mamba_block.c:0:0-0:0) | Orchestration, training loop, I/O |
| **Topologie** | `km_topology.c`, `wavefront_*.c` | ND indexing, strides, plans wavefront |
| **Puissance** | `kernels/*.c`, `cpu/*.asm`, `cuda/*.cu` | Compute pur (GEMM, scan, activations) |

---

### 5. Conventions de linkage

```c
// include/scan.h
void scan1d(ScanParams *p);           // Implémenté en ASM (cpu/scan1d.asm)
void scan1d_backward(ScanBackwardParams *p);  // Implémenté en C (cpu/scan1d_backward.c)

// include/scan_nd.h
int scannd(ScanNDParams *p);          // Référence C (src/scan_nd.c)
int om_scannd_forward(ScanNDParams *p); // CUDA (cuda/scan_nd.cu) - __CUDACC__ only
```

---

### 6. Points clés de l'intégration

- **Zero dependency**: Juste `gcc`, `nasm`, `libc`, `libm`
- **Inline kernels**: Pas de BLAS externe, tout en C pur
- **ASM hot paths**: [scan1d.asm](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/cpu/scan1d.asm:0:0-0:0) pour la récurrence SSM (forward)
- **Wavefront ND**: Topologie générique réutilisable par scanND et convND
- **Static library**: [libkmamba.a](cci:7://file:///home/samuel-yevi/Dev/OFFMODE/k-mamba/libkmamba.a:0:0-0:0) contient tout, linkage simple

Garde ceci pour refaire le readme