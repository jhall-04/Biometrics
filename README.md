### 🎯 Phase 1: Face Recognition

- [x] Implement face detection using MTCNN
- [x] Extract facial embeddings (FaceNet or ArcFace)
- [x] Create face enrollment & storage pipeline
- [x] Build matching system using cosine similarity
- [ ] Evaluate on LFW/VGGFace2 dataset
- [ ] Package as module: `face/recognition/`

---

### 🎯 Phase 2: Face Liveness Detection

- [ ] Train passive liveness classifier using CNN
- [ ] Add active liveness (blink/head motion via landmarks)
- [ ] Collect real vs spoof data samples
- [ ] Evaluate on CASIA-FASD or Replay-Attack
- [ ] Package as module: `face/liveness/`

---

### 🎯 Phase 3: Voice Recognition (Text-Dependent)

- [ ] Extract MFCC features from fixed passphrase
- [ ] Train speaker verification model (Siamese or GE2E)
- [ ] Build enrollment & verification interface
- [ ] Store and match voice embeddings
- [ ] Package as module: `voice/text_dependent/`

---

### 🎯 Phase 4: Voice Recognition (Text-Independent)

- [ ] Train speaker encoder on VoxCeleb dataset
- [ ] Add voice activity detection (VAD)
- [ ] Match voice samples regardless of content
- [ ] Evaluate with EER & ROC metrics
- [ ] Package as module: `voice/text_independent/`

---

### 🎯 Phase 5: System Integration

- [ ] Build REST API with FastAPI
- [ ] Add endpoints:
  - `/enroll/face`
  - `/enroll/voice`
  - `/verify/face`
  - `/verify/voice`
  - `/liveness`
- [ ] Implement score-level or decision-level fusion logic
- [ ] Add JSON-based session logging
- [ ] Package fusion logic in: `integration/fusion/`
- [ ] Optional: Add basic frontend (React.js or HTML5)