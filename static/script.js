document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const processBtn = document.getElementById('processBtn');
    const downloadBtn = document.getElementById('downloadBtn');

    let originalImage = null;

    // Обработка клика по области загрузки
    dropArea.addEventListener('click', () => fileInput.click());

    // Визуальные эффекты при drag&drop
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.style.borderColor = '#2ecc71';
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.style.borderColor = '#3498db';
    });

    // Загрузка файла
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.style.borderColor = '#3498db';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            updatePreview();
        }
    });

    fileInput.addEventListener('change', updatePreview);

    function updatePreview() {
        const file = fileInput.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage = e.target.result;
            imagePreview.src = originalImage;
            previewContainer.style.display = 'block';
            processBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Обработка изображения (заглушка)
    processBtn.addEventListener('click', () => {
        alert('Добавьте здесь свою логику обработки!');
        downloadBtn.disabled = false;
    });

    // Скачивание
    downloadBtn.addEventListener('click', () => {
        if (!originalImage) return;

        const link = document.createElement('a');
        link.download = 'processed-image.png';
        link.href = originalImage;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
