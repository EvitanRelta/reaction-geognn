

class GeoPredModel(nn.Layer):
    """tbd"""
    def __init__(self, model_config, compound_encoder):
        super(GeoPredModel, self).__init__()
        self.compound_encoder = compound_encoder

        self.hidden_size = model_config['hidden_size']
        self.dropout_rate = model_config['dropout_rate']
        self.act = model_config['act']
        self.pretrain_tasks = model_config['pretrain_tasks']

        # context mask
        if 'Cm' in self.pretrain_tasks:
            self.Cm_vocab = model_config['Cm_vocab']
            self.Cm_linear = nn.Linear(compound_encoder.embed_dim, self.Cm_vocab + 3)
            self.Cm_loss = nn.CrossEntropyLoss()
        # functinal group
        self.Fg_linear = nn.Linear(compound_encoder.embed_dim, model_config['Fg_size']) # 494
        self.Fg_loss = nn.BCEWithLogitsLoss()
        # bond angle with regression
        if 'Bar' in self.pretrain_tasks:
            self.Bar_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    act=self.act,
                    in_size=compound_encoder.embed_dim * 3,
                    out_size=1,
                    dropout_rate=self.dropout_rate)
            self.Bar_loss = nn.SmoothL1Loss()
        # bond length with regression
        if 'Blr' in self.pretrain_tasks:
            self.Blr_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    act=self.act,
                    in_size=compound_encoder.embed_dim * 2,
                    out_size=1,
                    dropout_rate=self.dropout_rate)
            self.Blr_loss = nn.SmoothL1Loss()
        # atom distance with classification
        if 'Adc' in self.pretrain_tasks:
            self.Adc_vocab = model_config['Adc_vocab']
            self.Adc_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    in_size=self.compound_encoder.embed_dim * 2,
                    act=self.act,
                    out_size=self.Adc_vocab + 3,
                    dropout_rate=self.dropout_rate)
            self.Adc_loss = nn.CrossEntropyLoss()

        print('[GeoPredModel] pretrain_tasks:%s' % str(self.pretrain_tasks))

    def _get_Cm_loss(self, feed_dict, node_repr):
        masked_node_repr = paddle.gather(node_repr, feed_dict['Cm_node_i'])
        logits = self.Cm_linear(masked_node_repr)
        loss = self.Cm_loss(logits, feed_dict['Cm_context_id'])
        return loss

    def _get_Fg_loss(self, feed_dict, graph_repr):
        fg_label = paddle.concat(
                [feed_dict['Fg_morgan'],
                feed_dict['Fg_daylight'],
                feed_dict['Fg_maccs']], 1)
        logits = self.Fg_linear(graph_repr)
        loss = self.Fg_loss(logits, fg_label)
        return loss

    def _get_Bar_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ba_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ba_node_j'])
        node_k_repr = paddle.gather(node_repr, feed_dict['Ba_node_k'])
        node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
        pred = self.Bar_mlp(node_ijk_repr)
        loss = self.Bar_loss(pred, feed_dict['Ba_bond_angle'] / np.pi)
        return loss

    def _get_Blr_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        pred = self.Blr_mlp(node_ij_repr)
        loss = self.Blr_loss(pred, feed_dict['Bl_bond_length'])
        return loss

    def _get_Adc_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ad_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ad_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        logits = self.Adc_mlp.forward(node_ij_repr)
        atom_dist = paddle.clip(feed_dict['Ad_atom_dist'], 0.0, 20.0)
        atom_dist_id = paddle.cast(atom_dist / 20.0 * self.Adc_vocab, 'int64')
        loss = self.Adc_loss(logits, atom_dist_id)
        return loss

    def forward(self, graph_dict, feed_dict, return_subloss=False):
        """
        Build the network.
        """
        node_repr, edge_repr, graph_repr = self.compound_encoder.forward(
                graph_dict['atom_bond_graph'], graph_dict['bond_angle_graph'])
        masked_node_repr, masked_edge_repr, masked_graph_repr = self.compound_encoder.forward(
                graph_dict['masked_atom_bond_graph'], graph_dict['masked_bond_angle_graph'])

        sub_losses = {}
        if 'Cm' in self.pretrain_tasks:
            sub_losses['Cm_loss'] = self._get_Cm_loss(feed_dict, node_repr)
            sub_losses['Cm_loss'] += self._get_Cm_loss(feed_dict, masked_node_repr)
        if 'Fg' in self.pretrain_tasks:
            sub_losses['Fg_loss'] = self._get_Fg_loss(feed_dict, graph_repr)
            sub_losses['Fg_loss'] += self._get_Fg_loss(feed_dict, masked_graph_repr)
        if 'Bar' in self.pretrain_tasks:
            sub_losses['Bar_loss'] = self._get_Bar_loss(feed_dict, node_repr)
            sub_losses['Bar_loss'] += self._get_Bar_loss(feed_dict, masked_node_repr)
        if 'Blr' in self.pretrain_tasks:
            sub_losses['Blr_loss'] = self._get_Blr_loss(feed_dict, node_repr)
            sub_losses['Blr_loss'] += self._get_Blr_loss(feed_dict, masked_node_repr)
        if 'Adc' in self.pretrain_tasks:
            sub_losses['Adc_loss'] = self._get_Adc_loss(feed_dict, node_repr)
            sub_losses['Adc_loss'] += self._get_Adc_loss(feed_dict, masked_node_repr)

        loss = 0
        for name in sub_losses:
            loss += sub_losses[name]
        if return_subloss:
            return loss, sub_losses
        else:
            return loss
