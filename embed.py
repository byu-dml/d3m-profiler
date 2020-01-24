import sent2vec, csv


if __name__ == '__main__':
    model = sent2vec.Sent2vecModel()
    model.load_model('pretrained_embedding_model.bin')
    vocab = model.get_vocabulary()
    uni_embs, vocab = model.get_unigram_embeddings()

    input = []
    with open('train_data.csv', 'r') as in_file:
        csv_reader = csv.DictReader(in_file)
        for row in csv_reader:
            input.append(row)

    output_headers = (
        ['datasetName_'+str(i) for i in range(700)] +
        ['description_'+str(i) for i in range(700)] +
        ['colName_'+str(i) for i in range(700)] +
        ['colType']
    )
    with open('train_data_embedded.csv', 'w') as out_file:
        csv_writer = csv.DictWriter(out_file, fieldnames=output_headers)
        csv_writer.writeheader()
        for in_row in input:
            out_row_values = (
                model.embed_sentence(in_row['datasetName'].lower()).tolist()[0] +
                model.embed_sentence(in_row['description'].lower()).tolist()[0] +
                model.embed_sentence(in_row['colName'].lower()).tolist()[0] +
                [in_row['colType']]
            )

            out_row = {output_headers[i]: out_row_values[i] for i in range(len(output_headers))}
            csv_writer.writerow(out_row)
