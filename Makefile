create-librimix:
	python data/LibriMix/create_librimix_from_metadata.py \
	--librispeech_dir data/LibriMix/LibriSpeech \
    --metadata_dir data/LibriMix/metadata/Libri2Mix \
    --librimix_outdir data/LibriMix/storage_dir \
    --n_src 2 \
    --freqs 8k \
    --modes min max \
    --types mix_clean