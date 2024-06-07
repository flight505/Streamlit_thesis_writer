# Function to create embeddings and ChromaDB
def create_chroma_db(md_folder, bib_folder, db_path):
    client = chromadb.Client()
    collection = client.create_collection("academic_papers")

    # Get the first .bib file from scr/BIBs and parse it
    bib_files = [f for f in os.listdir(bib_folder) if f.endswith(".bib")]
    if bib_files:
        with open(os.path.join(bib_folder, bib_files[0])) as bibtex_file:
            parser = BibTexParser()
            parser.customization = convert_to_unicode
            bib_database = bibtexparser.load(bibtex_file, parser=parser)

    for md_file in os.listdir(md_folder):
        if md_file.endswith(".md"):
            with open(os.path.join(md_folder, md_file), "r") as f:
                text = f.read()
            emb = embedding_function(text)
            collection.add({"content": text, "embedding": emb})

    client.save(db_path)
