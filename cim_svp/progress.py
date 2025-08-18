from tqdm.auto import tqdm
import joblib

class tqdm_joblib:
    """
    Context-manager to tie joblib.Parallel to a tqdm progress-bar.

    Usage
    -----
    with tqdm_joblib(tqdm(total=N)) as bar:
        Parallel(n_jobs=...)(...)
    """
    def __init__(self, tqdm_obj: tqdm):
        self.tqdm_obj = tqdm_obj
        self._old_cb  = None

    def __enter__(self):
        self._old_cb = joblib.parallel.BatchCompletionCallBack
        outer = self.tqdm_obj

        class _BatchCB(self._old_cb):
            def __call__(self, *a, **k):
                outer.update(n=self.batch_size)
                return super().__call__(*a, **k)

        joblib.parallel.BatchCompletionCallBack = _BatchCB
        return self.tqdm_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        joblib.parallel.BatchCompletionCallBack = self._old_cb
        self.tqdm_obj.close()
