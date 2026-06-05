import { FaExclamationTriangle } from "react-icons/fa";
import PropTypes from "prop-types";

export default function DeleteAccountModal({ isOpen, onClose, onConfirm, error, loading}) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
            
            <div className="relative w-full max-w-md bg-[#0b1e42] border border-red-500/30 p-6 rounded-2xl shadow-2xl z-10 animate-in fade-in zoom-in-95 duration-200 text-white">
                <h3 className="text-xl font-bold text-red-300 mb-2 flex items-center">
                    <FaExclamationTriangle className="mr-2" size={20} />
                    Hapus Akun Permanen?
                </h3>
                <p className="text-sm text-gray-300 mb-6 leading-relaxed">
                    Tindakan ini tidak dapat dibatalkan. Menekan tombol di bawah dapat menghapus seluruh data karier dan data diri Anda dari basis data KAVA.
                </p>

                {error && (
                    <div className="bg-red-500/2- border border-red-500 text-red-200 px-3 py-2 rounded-lg text-xs mb-4 text-center">
                        {error}
                    </div>
                )}

                <div className="flex justify-end space-x-3">
                    <button
                        onClick={onClose}
                        disabled={loading}
                        className="px-4 py-2 bg-transparent text-gray-400 hover:text-white text-sm font-medium"
                    >
                        Batal
                    </button>
                    <button
                        onClick={onConfirm}
                        disabled={loading}
                        className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded-xl text-sm font-medium text-white transition-colors"
                    >
                        {loading ? "Menghapus..." : "Ya, Hapus Permanen"}
                    </button>
                </div>
            </div>
        </div>
    );
}

DeleteAccountModal.propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    onConfirm: PropTypes.func.isRequired,
    error: PropTypes.string,
    loading: PropTypes.bool
};