/**
 * Manga Translator - Main JavaScript
 * Handles dropdowns, file uploads, and form interactions
 */

document.addEventListener("DOMContentLoaded", () => {
    initCustomSelects();
    initLocalStorage();
    initFileUpload();
});

/**
 * Initialize custom select dropdowns
 */
function initCustomSelects() {
    const selectWrappers = document.querySelectorAll('.select-wrapper');

    selectWrappers.forEach(wrapper => {
        const selectBox = wrapper.querySelector('.custom-select');
        if (!selectBox) return;

        const selectedText = selectBox.querySelector('.selected');
        const options = selectBox.querySelector('.options');
        const optionList = selectBox.querySelectorAll('.option');

        if (!optionList.length) return;

        // Set default
        const defaultOption = optionList[0];
        selectedText.textContent = defaultOption.textContent;
        defaultOption.classList.add('selected');

        // Toggle dropdown
        selectBox.addEventListener('click', (e) => {
            e.stopPropagation();
            
            // Close other dropdowns
            document.querySelectorAll('.custom-select.open').forEach(s => {
                if (s !== selectBox) {
                    s.classList.remove('open');
                    s.querySelector('.options').style.display = 'none';
                }
            });
            
            const isOpen = options.style.display === 'block';
            options.style.display = isOpen ? 'none' : 'block';
            selectBox.classList.toggle('open', !isOpen);
        });

        // Option selection
        optionList.forEach(option => {
            option.addEventListener('click', (e) => {
                e.stopPropagation();
                selectedText.textContent = option.textContent;
                optionList.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');

                // Handle special selects
                handleSelectChange(selectBox.id, option.textContent);

                // Save to localStorage
                if (selectBox.id) {
                    localStorage.setItem('select_' + selectBox.id, option.textContent);
                }
            });
        });

        // Close on outside click
        document.addEventListener('click', () => {
            options.style.display = 'none';
            selectBox.classList.remove('open');
        });

        // Restore from localStorage
        if (selectBox.id) {
            const savedValue = localStorage.getItem('select_' + selectBox.id);
            if (savedValue) {
                optionList.forEach(opt => {
                    if (opt.textContent === savedValue) {
                        selectedText.textContent = savedValue;
                        optionList.forEach(o => o.classList.remove('selected'));
                        opt.classList.add('selected');
                        handleSelectChange(selectBox.id, savedValue);
                    }
                });
            }
        }
    });
}

/**
 * Handle special select changes
 */
function handleSelectChange(selectId, value) {
    // Style select - show/hide custom prompt
    if (selectId === 'style') {
        const customWrapper = document.getElementById('custom-prompt-wrapper');
        if (customWrapper) {
            customWrapper.style.display = value.includes('Custom') ? 'block' : 'none';
        }
    }

    // Translator select - show/hide settings
    if (selectId === 'translator') {
        const copilotSettings = document.getElementById('copilot-settings');
        const geminiSettings = document.getElementById('gemini-settings');
        const openrouterSettings = document.getElementById('openrouter-settings');

        // Hide all first
        if (copilotSettings) copilotSettings.style.display = 'none';
        if (geminiSettings) geminiSettings.style.display = 'none';
        if (openrouterSettings) openrouterSettings.style.display = 'none';

        // Show relevant settings
        if (value.includes('Local LLM')) {
            if (copilotSettings) copilotSettings.style.display = 'block';
        } else if (value.includes('Gemini')) {
            if (geminiSettings) geminiSettings.style.display = 'block';
        } else if (value.includes('OpenRouter')) {
            if (openrouterSettings) openrouterSettings.style.display = 'block';
        }
    }

    // OpenRouter provider select - update hidden input
    if (selectId === 'provider_select') {
        const hiddenInput = document.getElementById('openrouter_provider');
        const selectedOption = document.querySelector('#provider_select .option.selected');
        if (hiddenInput && selectedOption) {
            const providerValue = selectedOption.getAttribute('data-value') || 'free';
            hiddenInput.value = providerValue;
            localStorage.setItem('openrouter_provider_value', providerValue);
            console.log('Provider changed to:', providerValue);
        }
    }
    }

/**
 * Initialize localStorage for inputs
 */
function initLocalStorage() {
    // Text inputs to save/restore
    const inputsToSave = [
        'gemini_api_key',
        'copilot_server',
        'copilot_model_input',
        'custom_prompt'
    ];

    inputsToSave.forEach(id => {
        const input = document.getElementById(id);
        if (!input) return;
        
        // Restore from localStorage (if input is empty)
        if (!input.value) {
            const saved = localStorage.getItem(id);
            if (saved) input.value = saved;
        }
        
        // Save on change
        input.addEventListener('input', () => {
            localStorage.setItem(id, input.value);
        });
    });

    // Checkboxes
    ['context_memory', 'detect_black_bubbles', 'split_long_images'].forEach(id => {
        const checkbox = document.getElementById(id);
        if (!checkbox) return;
        
        const saved = localStorage.getItem(id);
        if (saved !== null) checkbox.checked = saved === 'true';
        
        checkbox.addEventListener('change', () => {
            localStorage.setItem(id, checkbox.checked);
        });
    });
    
    // OpenRouter provider - restore from localStorage
    const savedProvider = localStorage.getItem('openrouter_provider_value');
    if (savedProvider) {
        const hiddenInput = document.getElementById('openrouter_provider');
        if (hiddenInput) hiddenInput.value = savedProvider;
        
        // Update dropdown display
        const providerSelect = document.querySelector('#provider_select');
        if (providerSelect) {
            const options = providerSelect.querySelectorAll('.option');
            const selectedSpan = providerSelect.querySelector('.selected');
            options.forEach(opt => {
                if (opt.getAttribute('data-value') === savedProvider) {
                    options.forEach(o => o.classList.remove('selected'));
                    opt.classList.add('selected');
                    if (selectedSpan) selectedSpan.textContent = opt.textContent;
                }
        });
    }
    }
}

/**
 * Initialize file upload handling
 */
function initFileUpload() {
const fileUpload = document.getElementById('file-upload');
    
if (fileUpload) {
        fileUpload.addEventListener('change', function() {
            handleFileSelection(this.files, 'file-text', 'file-list');
        });
    }
}

/**
 * Handle file selection display
 */
function handleFileSelection(files, textElementId, listElementId) {
    const fileText = document.getElementById(textElementId);
    const fileList = document.getElementById(listElementId);
    
    if (!fileText || !files.length) return;

        if (files.length === 0) {
        fileText.textContent = 'Kéo thả hoặc click để chọn ảnh';
        if (fileList) fileList.innerHTML = '';
            return;
        }

        if (files.length === 1) {
        fileText.textContent = truncateFileName(files[0].name, 30);
        if (fileList) fileList.innerHTML = '';
        } else {
            fileText.textContent = `📁 ${files.length} ảnh đã chọn`;

        if (fileList) {
            fileList.innerHTML = '';
            const maxShow = 5;
            
            for (let i = 0; i < Math.min(files.length, maxShow); i++) {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.textContent = truncateFileName(files[i].name, 35);
                fileList.appendChild(fileItem);
            }

            if (files.length > maxShow) {
                const moreItem = document.createElement('div');
                moreItem.className = 'file-item more';
                moreItem.textContent = `... và ${files.length - maxShow} ảnh khác`;
                fileList.appendChild(moreItem);
            }
        }
    }
}

/**
 * Truncate file name
 */
function truncateFileName(fileName, maxLength) {
    if (fileName.length <= maxLength) return fileName;
    
    const ext = fileName.split('.').pop();
    const nameWithoutExt = fileName.substring(0, fileName.length - ext.length - 1);
    const truncatedName = nameWithoutExt.substring(0, maxLength - ext.length - 4);
    
    return truncatedName + '...' + ext;
}

/**
 * Update hidden inputs before form submission
 */
function updateHiddenInputs() {
    const getSelectedText = (id) => {
        const el = document.querySelector(`#${id} .selected`);
        return el ? el.innerText : '';
    };

    // Update hidden fields
    const fields = ['source_lang', 'language', 'translator', 'style', 'font', 'ocr'];
    fields.forEach(field => {
        const hidden = document.getElementById('selected_' + field);
        if (hidden) {
            hidden.value = getSelectedText(field);
        }
    });

    // Validate Gemini API key (required)
    const translator = getSelectedText('translator');
    if (translator.includes('Gemini')) {
        const apiKey = document.getElementById('gemini_api_key');
        if (!apiKey?.value?.trim()) {
            alert('Vui lòng nhập Gemini API Key!');
            return false;
        }
    }
    // Note: OpenRouter has server-side fallback key, no strict validation needed

    // Check files
    const uploadMode = document.getElementById('upload_mode');
    const mode = uploadMode ? uploadMode.value : 'files';
    
    let hasFiles = false;
    if (mode === 'files') {
        const fileUpload = document.getElementById('file-upload');
        hasFiles = fileUpload && fileUpload.files.length > 0;
    } else if (mode === 'folder') {
        const folderUpload = document.getElementById('folder-upload');
        hasFiles = folderUpload && folderUpload.files.length > 0;
    } else if (mode === 'multi-folder') {
        const multiFolderUpload = document.getElementById('multi-folder-upload');
        hasFiles = multiFolderUpload && multiFolderUpload.files.length > 0;
    }

    if (!hasFiles) {
        alert('Vui lòng chọn ít nhất 1 ảnh hoặc thư mục!');
        return false;
    }

    // Show loading
    const form = document.querySelector('form');
    if (form) form.style.display = 'none';
    
    const loadingImg = document.getElementById('loading-img');
    const loadingP = document.getElementById('loading-p');
    if (loadingImg) loadingImg.style.display = 'block';
    if (loadingP) loadingP.style.display = 'block';

    return true;
}

/**
 * Natural sort comparison
 */
function naturalSort(a, b) {
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

/**
 * Format bytes to human readable
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
