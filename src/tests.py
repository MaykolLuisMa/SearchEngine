from pre_procesamiento import *
print("--------------------------------------------------------")

documents = get_dataset("cranfield")
documents = [Document(0,"la vida","the life is beautiful, but could be better"),
             Document(1,"rezar", "an experimental study of a wing in a propeller slipstream was made in order to determine the spanwise distribution")]
generate_corpus(documents)
corpus = get_corpus()
print("--------------------------------------------------------")
