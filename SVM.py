# Khởi tạo các thư viện cần thiết
import os, string, re
import glob
from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix





# Đường dẫn dữ liệu
path_train ='data/Train_Full'
path_test ='data/Test_Full'

# Các bước tiền xử lý văn bản

def normalText(sent):
    patURL = r"(?:http://|www.)[^\"]+"
    sent = re.sub(patURL,'website',sent)
    sent = re.sub('\.+','.',sent)
    sent = re.sub('\\s+',' ',sent)
    return sent

# Tách từ
def tokenizer(text):
    token = ViTokenizer.tokenize(text)
    return token

# Loại bỏ stopword trong tiếng Việt
with open("vi_stopwords.txt","r",encoding='utf8') as file:
    stopwords = file.read().split('\n')
def remove_stopword(text):
    text_new = " "
    for word in text.split(' '):
        if word.replace("_", " ").strip() not in stopwords:
            text_new +=  word +" "
    return text_new

# Hàm tiền xử lý
def clean_doc(doc):
    # Tách dấu câu ra khỏi chữ trước khi tách từ
    for punc in string.punctuation:
        doc = doc.replace(punc,' '+ punc + ' ')
    # Thay thế các link web = website, hagtag ~ hagtag
    doc = normalText(doc)
    # Tách từ đối với dữ liệu
    doc = tokenizer(doc)
    # Đưa tất cả về chữ thường
    doc = doc.lower()
    # Xóa nhiều khoảng trắng thành 1 khoảng trắng
    doc = re.sub(r"\?", " \? ", doc)
    # Thay thế các giá trị số thành ký tự num
    doc = re.sub(r"[0-9]+", " num ", doc)
    # Xóa bỏ các dấu câu không cần thiết
    for punc in string.punctuation:
        if punc !="_":
            doc = doc.replace(punc,' ')
    doc = re.sub('\\s+',' ',doc)
    return doc


# Hàm đọc dữ liệu
def read_data(folder_path,threshold=1000):
    documents = []
    labels = []
    #print(os.listdir(folder_path))
    for category in os.listdir(folder_path):
        count = 0 
        print(category)
        path_new = folder_path+ "/"+category + "/*.txt"
        for filename in glob.glob(path_new):
            if count <= threshold:
                with open(filename,'r',encoding="utf-16") as file:
                    content = file.read()
                    documents.append(content)
                    labels.append(category)
                    count +=1
            else:
                break
    return documents, labels

print("Dang doc du lieu tap Training....")
print("*" * 30)
X_train, y_train = read_data(path_train,100)

print("Dang doc du lieu tap Testing....")
print("*" * 30)

X_test, y_test = read_data(path_test,100)

print("So luong data trong tap train: ",len(X_train))
print("So luong data trong tap test: ",len(X_test))


# Danh sach cac nhãn trong dữ liệu
list_label = ["Chinh tri Xa hoi","Doi song","Khoa hoc","Kinh doanh","Phap luat","Suc khoe","The gioi","The thao","Van hoa","Vi tinh"]
print("So luong nhan la: ", len(list_label))


# Đoạn code này sẽ đọc dữ liệu trong tập train và chạy hàm tiền xử lý.
# Đối với nhãn của văn bản thì sẽ được chuyển thành dạng số với giá trị tương ứng theo index của list_label,
# Ví dụ như nhãn là "Chinh tri xa hoi" sẽ được chuyển thành 0, "Khoa hoc" là 2.
X_train_proceed = []
y_train_encoded = []
X_test_proceed = []
y_test_encoded = []
for index,data in enumerate(X_train):
    X_train_proceed.append(clean_doc(data))
    y = list_label.index(y_train[index])
    y_train_encoded.append([y])
    
for index,doc in enumerate(X_test):
    X_test_proceed.append(clean_doc(doc))
    y = list_label.index(y_test[index])
    y_test_encoded.append([y])

# Biểu diễn văn bản thành vector dựa theo chỉ số TF-IDF và đưa vào mô hình Linear SVM để huấn luyện
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),use_idf=True, smooth_idf=True)
X_train_tfidf = vectorizer.fit_transform(X_train_proceed)

print("Kich thuoc cua vector sau khi su dung TFIDF: ", X_train_tfidf.shape)


# Khởi tạo mô hình Linear SVM để huấn luyện mô hình trên tập huấn luyện
print("Bat dau khoi tao mo hinh SVM ....")
classifier = svm.LinearSVC(multi_class="crammer_singer", C = 0.1,  penalty='l2', random_state= 42, verbose=1)
classifier.fit(X_train_tfidf, y_train_encoded)
print("Mo hinh da huan luyen xong !!")

# Đoạn code này sẽ chuyển các dữ liệu trong tập test thành các vector tf-idf sau đó dùng mô hình classifier để dựa đoán
# Sau đó sử dụng các độ đo Accuracy, F1-score, Precision, Recall để đánh giá hiệu quả mô hình


X_test_tfidf = vectorizer.transform(X_test_proceed)
# Dưa vector tf-idf trong tập test vào mô hình dự đoán
predicted = classifier.predict(X_test_tfidf)
y_pred = []
for y in predicted:
    y_pred.append([int(y)])
    
print("Ket qua cua phuong phap SVM")
print("Accuracy: ",accuracy_score(y_test_encoded, y_pred))
print("F1 - score: ",f1_score(y_test_encoded, y_pred, average="macro"))
print("Precision: ",precision_score(y_test_encoded, y_pred, average="macro"))
print("Recall: ",recall_score(y_test_encoded, y_pred, average="macro"))

print("Ket qua chi tiet tung nhan")
print(classification_report(y_test_encoded,y_pred,target_names=list_label))


