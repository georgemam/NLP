NLP Project 2025

Υλοποίηση της εργασίας για το μάθημα Επεξεργασία Φυσικής Γλώσσας 2025. Το έργο επικεντρώνεται στη μετατροπή μη δομημένων κειμένων σε σαφή, σωστά και καλά οργανωμένα, χρησιμοποιώντας τεχνικές NLP. Τα αποτελέσματα αναλύονται με βάση τη σημασιολογική ομοιότητα (cosine similarity) και οπτικοποιούνται μέσω word embeddings με PCA.

Για την εκτέλεση, ακολουθήστε τα παρακάτω βήματα.

Προαπαιτούμενα

Python >= 3.11
Conda
Poetry

git clone https://github.com/DIONISISPX/nlp2025.git
cd nlp
conda create -n nlp python=3.11
conda activate nlp
poetry install


Μπορείτε να εκτελέσετε τα πειράματα χρησιμοποιώντας τις παρακάτω εντολές από τον κύριο φάκελο του έργου ή με το περιβάλλον που θέλετε

poetry run python source/A.py
poetry run python source/B.py
poetry run python source/analysis.py

Η αναλυτική συζήτηση των αποτελεσμάτων και των ευρημάτων περιλαμβάνεται στο Report.pdf