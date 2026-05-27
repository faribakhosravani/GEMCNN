import torch
import torch.nn as nn
from .encoders import SpectralCNNEncoder, ElementalEncoder
from .fusion   import FusionHead


class GEMCNN(nn.Module):
    """
    Gemtelligence multi-modal classifier  (Schmetzer et al., 2023).

    Tasks
    -----
    'od' → 4-class origin determination  (Kashmir / Burma / Sri Lanka / Madagascar)
    'td' → 2-class treatment detection   (treated / non-treated)

    Inputs
    ------
    uv    : (B, 2, L_uv)        two perpendicular UV spectra
    ftir  : (B, 1, L_ftir)      single FTIR spectrum
    xrf   : (B, F_xrf)          XRF elemental features
    icpms : (B, F_icp) | None   ICP-MS features (optional; TD task omits these)
    """

    TASK_CLASSES = {'od': 4, 'td': 2}

    def __init__(self,
                 uv_input_len:      int   = 1200,
                 ftir_input_len:    int   = 1700,
                 xrf_features:      int   = 30,
                 icpms_features:    int   = 40,
                 # CNN encoder hyper-params
                 cnn_hidden_dim:    int   = 128,
                 cnn_kernel_size:   int   = 17,
                 cnn_stride:        int   = 2,
                 cnn_num_blocks:    int   = 6,
                 # Elemental encoder hyper-params
                 elem_hidden_dim:   int   = 32,
                 elem_num_blocks:   int   = 2,
                 elem_num_heads:    int   = 4,
                 elem_ff_dim:       int   = 128,
                 elem_dropout:      float = 0.1,
                 # Task
                 task:              str   = 'od'):
        super().__init__()

        if task not in self.TASK_CLASSES:
            raise ValueError(f"task must be one of {list(self.TASK_CLASSES)}")

        self.task          = task
        self.icpms_features = icpms_features
        num_classes        = self.TASK_CLASSES[task]
        total_elemental    = xrf_features + icpms_features

        # ── encoders ──────────────────────────────────────────────────
        self.uv_encoder = SpectralCNNEncoder(
            in_channels=2,
            hidden_dim=cnn_hidden_dim,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            num_blocks=cnn_num_blocks,
        )
        self.ftir_encoder = SpectralCNNEncoder(
            in_channels=1,
            hidden_dim=cnn_hidden_dim,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            num_blocks=cnn_num_blocks,
        )
        self.elemental_encoder = ElementalEncoder(
            num_features=total_elemental,
            hidden_dim=elem_hidden_dim,
            num_blocks=elem_num_blocks,
            num_heads=elem_num_heads,
            ff_dim=elem_ff_dim,
            dropout=elem_dropout,
        )

        # ── compute fusion dim via dry run ─────────────────────────────
        fusion_dim = self._fusion_dim(uv_input_len, ftir_input_len, total_elemental)

        # ── fusion head ───────────────────────────────────────────────
        self.fusion = FusionHead(fusion_dim, num_classes)

    # ------------------------------------------------------------------
    def _fusion_dim(self, uv_len: int, ftir_len: int, elem_feat: int) -> int:
        with torch.no_grad():
            uv_dim   = self.uv_encoder(torch.zeros(1, 2, uv_len)).shape[1]
            ftir_dim = self.ftir_encoder(torch.zeros(1, 1, ftir_len)).shape[1]
            elem_dim = self.elemental_encoder(torch.zeros(1, elem_feat)).shape[1]
        return uv_dim + ftir_dim + elem_dim

    # ------------------------------------------------------------------
    def forward(self,
                uv:    torch.Tensor,
                ftir:  torch.Tensor,
                xrf:   torch.Tensor,
                icpms: torch.Tensor | None = None) -> torch.Tensor:

        if icpms is None:
            icpms = torch.zeros(xrf.shape[0], self.icpms_features,
                                device=xrf.device, dtype=xrf.dtype)

        elem_in  = torch.cat([xrf, icpms], dim=1)

        uv_emb   = self.uv_encoder(uv)
        ftir_emb = self.ftir_encoder(ftir)
        elem_emb = self.elemental_encoder(elem_in)

        return self.fusion(uv_emb, ftir_emb, elem_emb)
