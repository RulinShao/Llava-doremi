#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


TASK_LIST = ['MNIST-M+number_recognition', 'LSUN+Image_Classification', 'LOC_NARRATIVES+coco_images_caption', 'Winoground+image_captioning', 'NOCAPS+image_caption', 'iconqa+fill_in_blank', 'VQA-E+image_captioning', 'DTD+coarse_grained_texture_classification', 'DVQA+charts_question_answer_2', 'Office_31+Image_Classification_Category', 'ImageNet-C+image_classification_weather', 'VQA_activity_recognition', 'CONCADIA+image_caption_context_1', 'DeepWeeds+weed_species_recognition', 'PlotQA+visual_question_answering', 'DVQA+charts_question_answer_1', '300w+human_portrait_classification', 'LFW+face_recognition', 'wikihow_image_text_step_order', 'HICO+human_activity_detection', 'wikihow_immediate_next_step_selection', 'PACS+guitar_image_category_classification', 'visdial+answer_question_9', 'ImageNet-C+image_classification_noise', 'coco+image_classification_appliance', 'Cars+car_brand_classification', 'LOC_NARRATIVES+ade20k_images_caption', 'FLICKR30K+caption_image', 'CrisisMMD+damage_severity_classification', 'visdial+answer_question_6', 'Yoga-82+yoga_pose_recognition', 'expw+expression_detection', 'VQAv2', 'VQA_sentiment_understanding', 'visdial+answer_question_2', 'crowdhuman+count_numbers', 'CLEVR_CoGenT+Multiple_Question_Answering', 'Total-Text+text_detection', 'DOMAIN_NET+painting_image_classification', 'ExDark+object_recognition', 'FlickrLogos-27+logo_detection', 'KVQA+image_captioning', 'VQA_positional_reasoning', 'ok_vqa', 'Office_31+Image_Classification_ObjectAndCategory', 'JHU-CROWD+scene_type_recognition', 'VQG+caption_generation', 'Clevr+Question_Answering', 'MEMOTION+sentiment_detection', 'PACS+dog_image_category_classification', 'CLEVR_CoGenT+Question_Answer_Matching', 'PACS+elephant_image_category_classification', 'coco+image_classification_vehicle', 'DOMAIN_NET+infograph_image_classification', 'FGVC_Aircraft+Aircraft_Classification_Manufacturer', 'semart+image_technique', 'SentiCap+image_sentiment_captioning', 'CrisisMMD+Informativeness_classification', 'PACS+horse_image_category_classification', 'wikihow_text_image_step_order', 'RAVEN+next_pattern', 'ImageNet-A+image_classification', 'SKETCH+living_organism_detection', 'cinic-10+image_classification_transport', 'Core50+Object_detection', 'CONCADIA+image_caption_context_2', 'semart+image_type', 'CrisisMMD+humanitarian_categories_classification', 'PACS+person_image_category_classification', 'FFHQ-Text+text-to-face_generation', 'coco+image_classification_sports', 'NABirds+body_part_detection', 'spot-the-diff+image_diff_identification', 'DOMAIN_NET+image_category_classification', 'PACS+giraffe_image_category_classification', 'KVQA+image_question_answer', 'cinic-10+object_presence_animal', 'Clevr+Multiple_Question_Answering', 'fairface+image_classification_gender', 'trainSet+image_classification', 'GQA', 'infographicvqa+single_document_question', 'FoodLogoDet-1500+food_logo_recognition', 'VQA_object_recognition', 'Caltech-256+image_classification', 'PACS+photo_object_classification', 'CLEVR+question_answer', 'VisDA-2017+image_classification', 'A-OKVQA+rationales_generation', 'Cars+car_classification', 'coco+image_classification_kitchen', 'ConceptualCaptions+image_captioning', 'cinic-10+image_classification_shipping', 'ObjectNet+Object_classfication', 'FGVC_Aircraft+Aircraft_Classification_Variant', 'DAQUAR+object_question_answer', 'A-OKVQA+answer_rationales_matching', 'fairface+image_classification_age', 'cinic-10+object_presence_shipping', 'visualgenome_vqa', 'semart+image_timeframe', 'VIQUAE+question_answer', 'image_text_selection', 'INATURALIST+supercategory_classification', 'HICO+object_classification', 'ImageNet-R+image_classification', 'LOC_NARRATIVES+flickr30k_images_caption', 'VOC2007+object_detection', 'infographicvqa+question_answer', 'iconqa+choose_txt', 'wikihow_next_step', 'coco+image_classification_animal', 'NABirds+bird_species_detection', 'WIKIART+art_classification', 'AI2D+visual_question_answering', 'Clevr+VQA_context', 'ITM', 'SKETCH+object_detection', 'semart+image_school', 'LOC_NARRATIVES+open_images_caption', 'LAD+object_detection_details', 'VQA_object_presence', 'AID+aerial_scene_classification', 'STL-10+Image_Classification', 'Places205+Image_env_classification', 'INATURALIST+phylum_classification', 'RAF_DB+human_emotion_detection', 'CONCADIA+image_description', 'ayahoo_test_images+animal_object_vehicle_image_classification', 'VQA_color', 'DOMAIN_NET+quickdraw_image_classification', 'VQA_sport_recognition', 'CUB-200-2011+Bird_Classification', 'VQA_scene_recognition', 'FGVC_Aircraft+Aircraft_Classification', 'WIT+detailed_description', 'recipe-qa+visual_coherence', 'DOMAIN_NET+clipart_image_classification', 'CLEVR_CoGenT+Question_Answering', 'ImageNet-C+image_classification_general', 'VIZWIZ+image_captioning', 'Office_31+Image_Classification_Object', 'GTSRB+image_classification', 'model-vs-human+image_style_classification', 'PICKAPIC+image_short_description', 'CoVA+webpage_recognition', 'Dark-Zurich+time_classification', 'INATURALIST+genus_classification', 'VQARAD+question_answer', 'coco+image_classification_furniture', 'ImageNet-C+image_classification_blur', 'image_caption', 'MVTecAD+image_classification', 'NUS-WIDE+Animal_classification', 'CHART2TEXT+chart_caption', 'STANFORD_DOGS+dog_classification', 'semart+image_description', 'Caltech101+Image_classification', 'DOMAIN_NET+real_image_classification', 'image_quality', 'VisDA-2017+object_classification_train', 'REDCAPS+reddit_caption_2', 'INATURALIST+family_classification', 'DeepFashion_highres_Attribute_and_Category+Cloth_Classification', 'MemeCap+image_captioning', 'visdial+answer_question_4', 'A-OKVQA+visual_question_answering', 'STVQA+image_question_answer', 'GEOMETRY3K+geometry_question_answer', 'places365+Image_Classification', 'fairface+image_classification_race', 'DOCVQA+question_answer', 'Caltech101+Living_Thing_classification', 'MemeCap+meme_captioning', 'cinic-10+image_classification_animal', 'PACS+art_painting_object_classification', 'INATURALIST+order_classification', 'Clevr+Question_Answer_Matching', 'VisDA-2017+object_classification_validation', 'REDCAPS+reddit_caption_1', 'SCUT-CTW1500+text_detection', 'MVTecAD+anomaly_detection', 'PACS+cartoon_object_classification', 'VQA-E+visual_question_answering', 'CLEVR_CoGenT+VQA_context', 'multimodal_factual_checking', 'VQA_counting', 'question_image_match', 'DTD+all_texture_detection', 'textcaps+caption_generation', 'VQA_attribute', 'ImageNet-R+image_domain_classification', 'INATURALIST+class_classification', 'ImageNet-Sketch+image_classification', 'Office-Home+Image_classification', 'VQG+question_generation', 'vizwiz+question_answer', 'INATURALIST+latin_english_name_classification', 'VQA', 'FGVC_Aircraft+Aircraft_Classification_Family', 'VQA_utility_affordance', 'PACS+house_image_category_classification', 'FUNSD+text_detection', 'Road-Anomaly+anomaly_detection', 'Set5+image_recognition']
TASK_TO_IDX = {name: idx for idx, name in enumerate(TASK_LIST)}

class Doremi():
    def __init__(self,):
        self.reweight_eta = 1.0  # step size
        self.reweight_eps = 1e-4  # smoothing parameter
        self.accumulation_steps = 100  # frequency to update the parameters

        self.domain_list = TASK_LIST
        self.num_domains = len(self.domain_list)
        self.train_domain_weights = torch.ones(self.num_domains) / self.num_domains
        self.avg_train_domain_weights = torch.zeros_like(self.train_domain_weights)
        
        self.update_counter = 0
        self.perdomain_scores = []
        self.domain_ids = []

    def write_weights(self, weights):
        self.update_counter += 1
        self.train_domain_weights[:] = weights
        self.avg_train_domain_weights += weights
    
    def get_avg_weights(self,):
        return self.avg_train_domain_weights / self.update_counter
    
    def read_weights(self):
        return self.train_domain_weights.clone()

    def set_attributes(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update_domain_weights(self):
        # pdb.set_trace()
        train_domain_weights = self.read_weights()

        perdomain_scores = torch.zeros_like(self.train_domain_weights)

        # print(len(self.perdomain_scores))
        _perdomain_scores = torch.flatten(torch.cat(self.perdomain_scores, dim=0))
        _domain_ids = torch.flatten(torch.cat(self.domain_ids, dim=0))

        for domain_id in range(self.num_domains):
            indices = torch.where(_domain_ids == domain_id)
            # pdb.set_trace()
            if len(indices[0]) > 0:
                # no update if the domain isn't presented in current batch
                # another solution: consider sampling the batch according to domain weights
                perdomain_scores[domain_id] = torch.mean(_perdomain_scores[indices])
        
        # dist.all_reduce(perdomain_scores.contiguous(), op=ReduceOp.SUM)/torch.distributed.get_world_size
        # pdb.set_trace()
        # update domain weights
        log_new_train_domain_weights = torch.log(train_domain_weights) + self.reweight_eta * perdomain_scores
        
        # renormalize and smooth domain weights
        new_train_domain_weights = nn.functional.softmax(log_new_train_domain_weights, dim=0)
        train_domain_weights = (1-self.reweight_eps) * new_train_domain_weights + self.reweight_eps / len(new_train_domain_weights)
        
        self.write_weights(train_domain_weights)
    
    def get_train_domain_weights(self, domain_ids):
        train_domain_weights = self.train_domain_weights.to(domain_ids.device)
        train_domain_weights = train_domain_weights / train_domain_weights.sum()
        curr_domain_weights = train_domain_weights[domain_ids]
        return curr_domain_weights / sum(curr_domain_weights)
    
    def print_domain_weights(self):
        for domain, weight in zip(TASK_LIST, self.train_domain_weights):
            print(domain, weight)


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize doremi weights
        self.doremi = Doremi()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        domain_ids: Optional[int] = None,
        reference_loss: Optional[float] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)

            # compute loss without reduction
            # TODO: double check the shape
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.shape[0], -1)
            num_tokens = torch.where(shift_labels==-100, 0.0, 1.0)
            num_tokens = torch.sum(num_tokens, 1)

            # doremi requires pertoken loss
            loss = torch.sum(loss, 1) / num_tokens
            reference_loss = torch.sum(reference_loss, 1) / num_tokens
            excess_loss = torch.maximum(loss - reference_loss, torch.tensor(0))

            self.doremi.perdomain_scores.append(torch.flatten(excess_loss.detach()))
            self.doremi.domain_ids.append(torch.flatten(domain_ids.detach()))

            if len(self.doremi.perdomain_scores) == self.doremi.accumulation_steps:
                # update domain weights
                self.doremi.update_domain_weights()
                # doremi.print_domain_weights()
                if writer:
                    for domain, weight in zip(TASK_LIST, self.doremi.train_domain_weights):
                        writer.add_scalar(domain, weight, i)
                self.doremi.perdomain_scores = []
                self.doremi.domain_ids = []
            
            # compute the rescaled loss, divide by domain weights
            # assume doing uniform sampling
            curr_domain_weights = self.doremi.get_train_domain_weights(domain_ids)

            # get final loss
            loss = (loss * curr_domain_weights.detach()).sum()
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
