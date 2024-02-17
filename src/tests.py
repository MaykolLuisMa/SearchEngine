from pre_procesamiento import *
print("--------------------------------------------------------")
documents = get_dataset("cranfield")
documents = [Document("la vida","the life is beautiful, but could be better"),
             Document("rezar", "an experimental study of a wing in a propeller slipstream was made in order to determine the spanwise distribution")]
generate_corpus(documents)
repr, _ = get_corpus()
print("--------------------------------------------------------")
print(repr)