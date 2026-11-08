"""Retrieval index. Uses sentence-transformers when EMBED_MODEL is set and installed (use this for the thesis experiments);
otherwise TF-IDF, an offline fallback that is fine for tests but is NOT a substitute for embeddings in the final study."""
import os
import numpy as np

class Index:
    def __init__(self, texts, name=None):
        self.texts = list(texts); name = name if name is not None else os.getenv("EMBED_MODEL", ""); self.model = None; self.kind = "tfidf"
        if name:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(name); self.kind = f"sentence-transformers:{name}"
            except Exception as e: print(f"[embed] could not load '{name}' ({type(e).__name__}); using TF-IDF instead")
        if self.model: self.M = np.asarray(self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=False))
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            try: self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True); self.M = self.vec.fit_transform(self.texts)
            except ValueError: self.vec = TfidfVectorizer(sublinear_tf=True); self.M = self.vec.fit_transform(self.texts)
    def scores(self, q):
        if self.model: return self.M @ np.asarray(self.model.encode([q], normalize_embeddings=True, show_progress_bar=False))[0]
        return (self.M @ self.vec.transform([q]).T).toarray().ravel()
    def search(self, q, k=10):
        s = self.scores(q); idx = np.argsort(-s)[:k]; return [(int(i), float(s[i])) for i in idx if s[i] > 0]
    def sim_matrix(self):
        S = self.M @ self.M.T; return S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    @property
    def dedupe_threshold(self): return 0.85 if self.model else 0.6
