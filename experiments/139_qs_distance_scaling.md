# Exp139: QS Distance Scaling — Bacteria Shouting vs Human Shouting

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (6/6 checks, GPU-confirmed) |
| **Binary**     | `validate_qs_distance_scaling` |
| **Date**       | 2026-02-23 |
| **Phase**      | 36c — Why Analysis |

## Hypothesis

Bacterial QS signals have characteristic diffusion lengths that can be
meaningfully compared to human communication distances when normalized
to body lengths.

## Design

- Compute AHL characteristic diffusion length from D and degradation rate
- Tabulate all bacterial and human communication modes in body lengths
- Use Anderson model to show how geometry limits effective propagation
- Apply to mixed systems (inoculant in rhizosphere)

## Key Results — The Distance Table

| Bacterial mode       | Range  | Body× | Human equivalent          | Human range |
|----------------------|--------|-------|---------------------------|-------------|
| Contact (T6SS)       | 0.5 µm | 0.5× | Touching someone          | ~1m         |
| Contact (CDI)        | 1 µm   | 1×   | Arm's reach handshake     | ~1.75m      |
| Membrane vesicles    | 5 µm   | 5×   | Conversation distance     | ~9m         |
| **QS (dense biofilm)** | **10 µm** | **10×** | **Speaking across a room** | **~18m** |
| **QS (loose biofilm)** | **100 µm** | **100×** | **SHOUTING across a field** | **~175m** |
| QS (liquid, max)     | 3908 µm | 3908× | Sight range (person)     | ~7km        |
| VOC (gas phase)      | 10000 µm | 10000× | Bonfire visibility     | ~17.5km     |

## Key Findings

1. **QS in biofilm ≈ human shouting**: 10-100 body lengths matches the
   57 body lengths of a human shout (100m). Bacteria are literally
   shouting through chemistry.
2. **QS in liquid ≈ human sight**: 3908 body lengths, but the Anderson
   model shows this range is useless at planktonic dilution — like
   shouting into fog.
3. **The Anderson transition IS the fog-to-field transition**: 2D biofilm
   = shouting into fog (signal dies); 3D biofilm = shouting in open air
   (signal carries to the horizon).
4. **For mixed inoculant systems**: seed coatings (3D aggregation at root)
   outperform broadcast (dispersed planktonic) because they create the
   local 3D structure needed for QS establishment.
