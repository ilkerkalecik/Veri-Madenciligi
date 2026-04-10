Projeyi çalıştırmak için: 

pip install numpy pandas matplotlib networkx scikit-learn openpyxl

daha sonra kodlarda değiştirilmesi gereken 2 bölüm var: 
  base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "shakira.csv")        //kulanılacak datasetin ismi 

BASE_SIZE = 25 				// kullanıcının dinlediği ilk şarkı sayısı. Graflar bu ilk şarkılara göre olulturuluyor.
    
Daha sonra ise python run dosyaAdı.py


python3 keras_autoencoder_recommender.py --file shakira.csv --base-size 25 --top-k 10
python3 pytorch_embedding_recommender.py --file shakira.csv --base-size 25 --top-k 10
