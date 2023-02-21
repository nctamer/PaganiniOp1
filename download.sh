#!/bin/bash
#SBATCH --job-name=autoData
#SBATCH -n 1
#SBATCH --mem 4000
#SBATCH -p medium                  # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written

module load libsndfile
module load RubberBand
module load FFmpeg
module load Miniconda3
eval "$(conda shell.bash hook)"
conda init bash
conda activate supervised

!yt-dlp --yes-playlist https://www.youtube.com/playlist?list=PLJTogM3HqLItI9ax9MWlyOeZeiuXPgEqF  -x --audio-format wav --audio-quality 0  -i -o "audio/No%(playlist_index)02d/AntalZalai_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
!yt-dlp --yes-playlist https://youtube.com/playlist?list=PLwLTvGAP00b-5A_pVbkxg8yk6g7anXik7  -x --audio-format wav --audio-quality 0  -i -o "audio/No%(playlist_index)02d/AlicanSuener_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
!yt-dlp --yes-playlist https://www.youtube.com/playlist?list=PLfXppYtrZclQ52OgzSFu2cxSSwZDseigJ  -x --audio-format wav --audio-quality 0  -i -o "audio/No%(playlist_index)02d/ItzhakPerlman_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
!yt-dlp --yes-playlist https://youtube.com/playlist?list=PL7UqW2VticVLXyKCpmNqH1Nwmi4xu3qSI  -x --audio-format wav --audio-quality 0  -i -o "audio/No%(playlist_index)02d/JuliaFischer_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
!yt-dlp --yes-playlist https://www.youtube.com/playlist?list=PLn2--CWCjouOuWOprzNkaYKJ0g7FFMbce  -x --audio-format wav --audio-quality 0  -i -o "audio/No%(playlist_index)02d/InmoYang_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
!yt-dlp --yes-playlist https://youtube.com/playlist?list=PLdkjwZMK_CIqA7ZUWb7KOLEKrwahwRtVI  -x --audio-format wav --audio-quality 0  -i -o "audio/No%(playlist_index)02d/SalvatoreAccardo_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
!yt-dlp --yes-playlist https://www.youtube.com/playlist?list=PLIK4KBELyc3BCeHp9G2HqloTUqAFt8AgB  -x --audio-format wav --audio-quality 0  -i -o "audio/No%(playlist_index)02d/IlyaKaler_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
