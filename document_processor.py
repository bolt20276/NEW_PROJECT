# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
# import tempfile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from config import Config

# class DocumentProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=Config.CHUNK_SIZE,
#             chunk_overlap=Config.CHUNK_OVERLAP
#         )

#     def process_file(self, file_content, ext):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
#             tmp_file.write(file_content)
#             return self._load_document(tmp_file.name)

#     def _load_document(self, file_path):
#         if file_path.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         else:
#             loader = Docx2txtLoader(file_path)
#         return loader.load()


from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def process_file(self, file_content, ext):
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_content)
            return self._load_document(tmp_file.name)

    def _load_document(self, file_path):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)
        return loader.load()



