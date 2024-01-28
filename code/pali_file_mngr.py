from pathlib import Path
from random import randint
import multiprocessing

from code.pali_text_file import TextFile
from code.pali_vectorizer import Vectorizer

class FileMngr:
    
    langs = ["pli"]

    original_dir= "original"
    stemmed_dir= "stemmed"
    vectors_dir= "vectors"

    def __init__(self,
                 n_buckets,
                 output_root_path,
                 ref_path,
                 threads=0,
                 ) -> None:
        self.root_path = Path(output_root_path)
        self.ref_path = Path(ref_path)
        self.n_buckets = n_buckets
        self.threads = threads
        self.stemmed_path = self.init_stemmed_path_dic()


    def init_stemmed_path_dic(self):
        stemmed_path = {}
        for lang in self.langs:
            stemmed_path[lang] = self.root_path / lang / self.stemmed_dir
            stemmed_path[lang].mkdir(exist_ok=True, parents=True)
        return stemmed_path


    def vectors_path(self, TextFile_obj):
        path: Path = self.root_path / TextFile_obj.lang / self.vectors_dir / (
            "folder" + self.bucket_number(TextFile_obj))
        path.mkdir(exist_ok=True, parents=True)
        return path / TextFile_obj.vectors_file

    def get_stemmed_files(self, lang) -> list[Path]:
        result = list(self.stemmed_path[lang].rglob(
            "*" + TextFile.stemmed_extention)) # "*.csv"
        return result

    def make_output_path(self, custorm_output_dir) -> None:
        if not custorm_output_dir:
            output_path = self.input_dir / self.vectors_dir
        output_path.mkdir(exist_ok=True)
        return output_path

    def pickle_words_df(self, TextFile_obj):
        file_path = self.vectors_path(TextFile_obj)
        print(f"Saving {TextFile_obj.name} as {file_path}")
        TextFile_obj.words_df.to_pickle(file_path)

    def bucket_number(self, TextFile_obj):
        if self.n_buckets > 0:
            return f'{hash(TextFile_obj.name) % self.n_buckets:04}'
        else:
            return ""
